"""
选题一：基于 ResNet18 和语音交互的物体识别 Demo
- 使用预训练 ResNet18 提取 512 维图像特征
- 录入物体（名称 + 多张图片），保存特征到特征库
- 识别时计算余弦相似度，检索特征库
- 相似度低于阈值时判断为未知物体，回退到 ImageNet 分类给出可能的类别
- 语音交互（浏览器 Web Speech API：语音输入物体名称 + 语音播报识别结果）
- 特征库持久化，重启后仍可使用

运行：python feature_app.py
访问：http://localhost:8081
"""

import io
import os
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_MODEL_PATH = os.path.join(BASE_DIR, "resnet18_feature.pt")
CLF_MODEL_PATH = os.path.join(BASE_DIR, "resnet18_quantized.pt")
LABELS_PATH = os.path.join(BASE_DIR, "imagenet_labels.json")
DB_PATH = os.path.join(BASE_DIR, "feature_db.pkl")
FEATURE_IMAGES_DIR = os.path.join(BASE_DIR, "feature_images")
os.makedirs(FEATURE_IMAGES_DIR, exist_ok=True)

# 相似度阈值：低于此值判定为未知物体
SIMILARITY_THRESHOLD = 0.75

app = Flask(__name__)
# 改模板后自动重载（Debug off 时 Jinja2 默认不重载，需显式开启）
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ---------------------------------------------------------------------------
# 预处理（与 ImageNet 训练时一致）
# ---------------------------------------------------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# 加载特征提取模型
# ---------------------------------------------------------------------------
if not os.path.exists(FEATURE_MODEL_PATH):
    raise FileNotFoundError(
        f"未找到特征提取模型 {FEATURE_MODEL_PATH}，请先运行: python prepare_model.py"
    )

print(f"[加载] 特征提取模型 {FEATURE_MODEL_PATH} ...")
feature_model = torch.jit.load(FEATURE_MODEL_PATH, map_location="cpu")
feature_model.eval()

# 加载 ImageNet 分类模型（用于未知物体回退分类）
print(f"[加载] ImageNet 分类模型 {CLF_MODEL_PATH} ...")
clf_model = torch.jit.load(CLF_MODEL_PATH, map_location="cpu")
clf_model.eval()

# 加载类别标签
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    idx_to_label = {int(k): v for k, v in json.load(f).items()}
print(f"[加载] ImageNet 类别标签 {len(idx_to_label)} 类")

torch.set_num_threads(4)

# 预热
with torch.no_grad():
    _ = feature_model(torch.randn(1, 3, 224, 224))
    _ = clf_model(torch.randn(1, 3, 224, 224))
print("[加载] 模型预热完成（特征模型 512 维 + 分类模型 1000 类）")


# ---------------------------------------------------------------------------
# 特征库管理
# ---------------------------------------------------------------------------
class FeatureDatabase:
    """物体特征库：持久化存储，支持录入、检索、删除。"""

    def __init__(self, path):
        self.path = path
        self.data = {}  # {物体名: {"features": [np.array, ...], "images": [str, ...], "count": int}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)
            # 兼容旧数据：没有 images 字段时补空列表
            for v in self.data.values():
                if "images" not in v:
                    v["images"] = []
            total = sum(len(v["features"]) for v in self.data.values())
            print(f"[特征库] 已加载 {len(self.data)} 个物体，共 {total} 条特征")

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def add_object(self, name, features_list, image_filenames=None):
        """为物体添加多条特征及对应图片文件名。"""
        if name not in self.data:
            self.data[name] = {"features": [], "images": [], "count": 0}
        if "images" not in self.data[name]:
            self.data[name]["images"] = []
        self.data[name]["features"].extend(features_list)
        if image_filenames:
            self.data[name]["images"].extend(image_filenames)
        self.data[name]["count"] = len(self.data[name]["features"])
        self.save()

    def get_all_names(self):
        return list(self.data.keys())

    def get_info(self):
        """返回特征库摘要信息，包含所有图片 URL。"""
        info = {}
        for name, v in self.data.items():
            safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "object"
            img_dir = os.path.join(FEATURE_IMAGES_DIR, safe_name)
            images = v.get("images", [])
            # 返回每张图片的真实索引，避免按过滤后的位置删除导致索引错位
            image_items = []
            for idx, fname in enumerate(images):
                if os.path.exists(os.path.join(img_dir, fname)):
                    image_items.append({
                        "url": f"/feature_images/{safe_name}/{fname}",
                        "index": idx,
                    })
            info[name] = {
                "image_count": v["count"],
                "images": image_items,
                "image_urls": [it["url"] for it in image_items],
            }
        return info

    def delete_image(self, name, index):
        """删除物体的第 index 张图片及其对应特征。"""
        if name not in self.data:
            return False
        v = self.data[name]
        images = v.get("images", [])
        if index < 0 or index >= len(images):
            return False
        fname = images[index]
        # 删除图片文件
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "object"
        img_path = os.path.join(FEATURE_IMAGES_DIR, safe_name, fname)
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass
        # 删除对应特征与图片记录
        del images[index]
        if index < len(v["features"]):
            del v["features"][index]
        v["count"] = len(v["features"])
        self.save()
        return True

    def delete_object(self, name):
        if name in self.data:
            del self.data[name]
            self.save()
            # 同时删除该物体的所有图片目录
            safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "object"
            img_dir = os.path.join(FEATURE_IMAGES_DIR, safe_name)
            if os.path.isdir(img_dir):
                try:
                    import shutil
                    shutil.rmtree(img_dir)
                except Exception:
                    pass
            # 兼容旧版单文件
            old_img = os.path.join(FEATURE_IMAGES_DIR, f"{safe_name}.jpg")
            if os.path.exists(old_img):
                try:
                    os.remove(old_img)
                except Exception:
                    pass
            return True
        return False

    def search(self, query_feature, top_k=3):
        """
        在特征库中检索与查询特征最相似的物体。
        返回 [(物体名, 相似度), ...]，按相似度降序排列。
        """
        results = []
        for name, info in self.data.items():
            # 取该物体所有特征中与查询特征的最大相似度
            max_sim = 0.0
            for feat in info["features"]:
                sim = float(F.cosine_similarity(
                    query_feature, torch.from_numpy(feat).unsqueeze(0), dim=1
                ).item())
                if sim > max_sim:
                    max_sim = sim
            # 也计算平均相似度作为参考
            avg_sim = float(np.mean([
                F.cosine_similarity(
                    query_feature, torch.from_numpy(f).unsqueeze(0), dim=1
                ).item()
                for f in info["features"]
            ]))
            results.append({
                "name": name,
                "max_similarity": max_sim,
                "avg_similarity": avg_sim,
                "image_count": info["count"],
            })

        results.sort(key=lambda x: x["max_similarity"], reverse=True)
        return results[:top_k]


db = FeatureDatabase(DB_PATH)


# ---------------------------------------------------------------------------
# 特征提取辅助函数
# ---------------------------------------------------------------------------
def extract_feature(image_bytes):
    """从图片字节流提取 L2 归一化的 512 维特征向量。"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = feature_model(input_tensor)  # (1, 512)
        # L2 归一化
        return F.normalize(feat, p=2, dim=1)
    finally:
        img.close()


def features_to_numpy(feat_tensor):
    """将特征张量转为 numpy 数组。"""
    return feat_tensor.squeeze(0).cpu().numpy()


def classify_imagenet(image_bytes, top_k=3):
    """使用 ImageNet 分类模型对图片进行分类，返回 top-k 结果。"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = clf_model(input_tensor)
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            top_prob, top_catid = torch.topk(probs, top_k)
        results = []
        for p, c in zip(top_prob, top_catid):
            label = idx_to_label.get(c.item(), "未知")
            results.append({
                "name": label,
                "confidence": round(p.item() * 100, 2),
            })
        return results
    finally:
        img.close()


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect("/manage")


@app.route("/manage")
def manage():
    return render_template("feature_index.html")


@app.route("/register", methods=["POST"])
def register():
    """录入物体：接收物体名称和多张图片，提取特征并保存。"""
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "请输入物体名称"})

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"success": False, "message": "请至少上传一张图片"})

    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "object"
    obj_img_dir = os.path.join(FEATURE_IMAGES_DIR, safe_name)
    os.makedirs(obj_img_dir, exist_ok=True)

    # 取已有图片编号的最大值 +1 作为新图片起始编号，避免删除中间图片后覆盖已有文件
    existing_imgs = [f for f in os.listdir(obj_img_dir) if f.lower().endswith(".jpg")] if os.path.exists(obj_img_dir) else []
    start_idx = 0
    for fname in existing_imgs:
        try:
            start_idx = max(start_idx, int(os.path.splitext(fname)[0]) + 1)
        except ValueError:
            continue

    features_list = []
    image_filenames = []
    saved_count = 0
    for f in files:
        if f.filename == "":
            continue
        try:
            img_bytes = f.read()
            feat_tensor = extract_feature(img_bytes)
            features_list.append(features_to_numpy(feat_tensor))
            # 保存图片到物体子目录，按索引命名
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.thumbnail((320, 320))
            fname = f"{start_idx + saved_count}.jpg"
            img.save(os.path.join(obj_img_dir, fname), "JPEG", quality=85)
            image_filenames.append(fname)
            saved_count += 1
        except Exception as e:
            return jsonify({"success": False, "message": f"图片处理失败: {e}"})

    if features_list:
        db.add_object(name, features_list, image_filenames)
        total = db.data[name]["count"]
        return jsonify({
            "success": True,
            "message": f"物体「{name}」录入成功！本次添加 {saved_count} 张，"
                      f"累计 {total} 张图片特征。",
            "object_name": name,
            "total_images": total,
        })

    return jsonify({"success": False, "message": "没有有效的图片"})


@app.route("/recognize", methods=["POST"])
def recognize():
    """识别物体：上传一张图片，检索特征库返回最相似物体。"""
    if "image" not in request.files:
        return jsonify({"success": False, "message": "请上传图片"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "请上传图片"})

    try:
        image_bytes = file.read()
        feat_tensor = extract_feature(image_bytes)
    except Exception as e:
        return jsonify({"success": False, "message": f"图片处理失败: {e}"})

    # 转为 base64 预览
    import base64
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preview_buf = io.BytesIO()
    img.save(preview_buf, format="JPEG")
    preview_b64 = base64.b64encode(preview_buf.getvalue()).decode("ascii")
    img.close()

    if len(db.data) == 0:
        # 特征库为空，直接用 ImageNet 分类
        imagenet_results = classify_imagenet(image_bytes, top_k=3)
        best_imagenet = imagenet_results[0]
        return jsonify({
            "success": True,
            "recognized": False,
            "object_name": f"可能是：{best_imagenet['name']}",
            "similarity": 0,
            "threshold": round(SIMILARITY_THRESHOLD * 100, 2),
            "message": f"特征库为空，ImageNet 分类结果：{best_imagenet['name']}"
                      f"（{best_imagenet['confidence']:.1f}%）",
            "imagenet_top3": imagenet_results,
            "preview_b64": preview_b64,
        })

    results = db.search(feat_tensor, top_k=3)

    best = results[0]
    is_known = best["max_similarity"] >= SIMILARITY_THRESHOLD

    if is_known:
        # 特征库命中
        return jsonify({
            "success": True,
            "recognized": True,
            "object_name": best["name"],
            "similarity": round(best["max_similarity"] * 100, 2),
            "threshold": round(SIMILARITY_THRESHOLD * 100, 2),
            "message": f"识别结果：{best['name']}（相似度 {best['max_similarity']*100:.1f}%）",
            "top3": [
                {"name": r["name"], "similarity": round(r["max_similarity"] * 100, 2)}
                for r in results
            ],
            "preview_b64": preview_b64,
        })
    else:
        # 特征库未命中，回退到 ImageNet 分类
        imagenet_results = classify_imagenet(image_bytes, top_k=3)
        best_imagenet = imagenet_results[0]
        return jsonify({
            "success": True,
            "recognized": False,
            "object_name": f"可能是：{best_imagenet['name']}",
            "similarity": round(best["max_similarity"] * 100, 2),
            "threshold": round(SIMILARITY_THRESHOLD * 100, 2),
            "message": (
                f"未录入的物体（最高相似度 {best['max_similarity']*100:.1f}%）。"
                f"ImageNet 分类：{best_imagenet['name']}"
                f"（{best_imagenet['confidence']:.1f}%）"
            ),
            "top3": [
                {"name": r["name"], "similarity": round(r["max_similarity"] * 100, 2)}
                for r in results
            ],
            "imagenet_top3": imagenet_results,
            "preview_b64": preview_b64,
        })


@app.route("/database")
def database():
    """查看特征库。"""
    info = db.get_info()
    return jsonify({
        "objects": info,
        "total_objects": len(info),
        "total_features": sum(v["image_count"] for v in info.values()),
        "threshold": SIMILARITY_THRESHOLD,
    })


@app.route("/feature_images/<path:filename>")
def serve_feature_image(filename):
    """提供特征库物体的代表图。"""
    return send_from_directory(FEATURE_IMAGES_DIR, filename)


@app.route("/delete", methods=["POST"])
def delete():
    """删除特征库中的物体。"""
    name = request.form.get("name", "").strip()
    if not name and request.is_json:
        name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "message": "请提供物体名称"})
    if db.delete_object(name):
        return jsonify({"success": True, "message": f"已删除物体「{name}」"})
    return jsonify({"success": False, "message": f"未找到物体「{name}」"})


@app.route("/delete_image", methods=["POST"])
def delete_image():
    """删除物体的单张图片及其对应特征。"""
    name = request.form.get("name", "").strip()
    index_str = request.form.get("index", "").strip()
    if not name:
        return jsonify({"success": False, "message": "请提供物体名称"})
    try:
        index = int(index_str)
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "无效的图片索引"})
    if db.delete_image(name, index):
        return jsonify({"success": True, "message": "图片已删除"})
    return jsonify({"success": False, "message": "未找到该图片"})


@app.route("/clear", methods=["POST"])
def clear():
    """清空整个特征库。"""
    count = len(db.data)
    db.data.clear()
    db.save()
    # 同时删除所有物体图片目录及旧版单图
    import shutil
    for fn in os.listdir(FEATURE_IMAGES_DIR):
        full = os.path.join(FEATURE_IMAGES_DIR, fn)
        try:
            if os.path.isdir(full):
                shutil.rmtree(full)
            elif fn.lower().endswith(".jpg"):
                os.remove(full)
        except Exception:
            pass
    return jsonify({"success": True, "message": f"已清空特征库（共 {count} 个物体）"})


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": "resnet18_feature",
        "feature_dim": 512,
        "objects": len(db.data),
        "threshold": SIMILARITY_THRESHOLD,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
