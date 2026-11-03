import io
import json
import os

import torch
from PIL import Image
import torchvision.transforms as T
from flask import (
    Flask,
    request,
    render_template,
    flash,
    redirect,
    url_for,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet18_quantized.pt")
LABELS_PATH = os.path.join(BASE_DIR, "imagenet_labels.json")
ZH_LABELS_PATH = os.path.join(BASE_DIR, "imagenet_labels_zh.json")

# 允许上传的图片扩展名
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-me")

# ---------------------------------------------------------------------------
# 预处理 pipeline（与 ImageNet 训练时一致）
# ---------------------------------------------------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# 加载模型与标签
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"未找到量化模型 {MODEL_PATH}，请先运行: python prepare_model.py"
    )
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"未找到类别标签 {LABELS_PATH}，请先运行: python prepare_model.py"
    )

print(f"[加载] 模型 {MODEL_PATH} ...")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    idx_to_label = {int(k): v for k, v in json.load(f).items()}

# 中文标签（可选，不存在则只显示英文）
idx_to_label_zh = {}
if os.path.exists(ZH_LABELS_PATH):
    with open(ZH_LABELS_PATH, "r", encoding="utf-8") as f:
        idx_to_label_zh = {int(k): v for k, v in json.load(f).items()}
    print(f"[加载] 中文标签 {len(idx_to_label_zh)} 个")
else:
    print("[加载] 未找到中文标签文件，仅显示英文")

# CPU 多线程推理
torch.set_num_threads(4)

# 启动时预热一次前向传播，避免冷启动延迟
with torch.no_grad():
    _ = model(torch.randn(1, 3, 224, 224))
print("[加载] 模型预热完成")


def allowed_file(filename):
    """检查上传文件扩展名是否合法。"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    """主页：上传图片并返回识别结果。"""
    if request.method == "POST":
        # 检查是否有文件
        if "image" not in request.files:
            flash("请选择要上传的图片")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("请选择要上传的图片")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash(f"不支持的图片格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(request.url)

        # 读取图片字节（避免 stream 被 PIL 持有导致无法 close）
        file_bytes = file.read()
        img = None
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            input_tensor = transform(img).unsqueeze(0)  # 添加 batch 维度

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits[0], dim=0)
                top3_prob, top3_catid = torch.topk(probs, 3)

            results = [
                {
                    "label": idx_to_label.get(catid.item(), "unknown"),
                    "label_zh": idx_to_label_zh.get(catid.item(), ""),
                    "score": prob.item(),
                }
                for prob, catid in zip(top3_prob, top3_catid)
            ]

            # 把上传的图片转为 base64 以便在结果页预览
            import base64

            preview_buf = io.BytesIO()
            img.save(preview_buf, format="JPEG")
            preview_b64 = base64.b64encode(preview_buf.getvalue()).decode("ascii")

            return render_template(
                "result.html",
                results=results,
                preview_b64=preview_b64,
                filename=file.filename,
            )
        except Exception as e:  # noqa: BLE001
            flash(f"识别失败: {e}")
            return redirect(request.url)
        finally:
            if img is not None:
                img.close()

    return render_template("upload.html")


@app.route("/health")
def health():
    """健康检查端点（供 Docker / 负载均衡使用）。"""
    return {"status": "ok", "model": "resnet18", "classes": len(idx_to_label)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
