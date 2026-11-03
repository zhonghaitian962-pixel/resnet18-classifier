"""
ResNet-18 模型准备脚本
- 下载官方预训练 ResNet-18
- 下载 ImageNet 1000 类别索引
- 动态量化（INT8）减少内存、提升推理速度
- 转换为 TorchScript 以便独立部署
- 保存量化模型与类别标签文件

运行：python prepare_model.py
完成后会生成：
  - resnet18_quantized.pt      （量化 + TorchScript 模型）
  - imagenet_labels.json       （索引 -> 类别名称映射）
"""

import json
import os
import urllib.request

import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ImageNet class index 文件（格式 {"0": ["n01440764", "tench"], ...}）
# 多个备用源，依次尝试
CLASS_INDEX_URLS = [
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
    "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json",
]

MODEL_OUT = os.path.join(BASE_DIR, "resnet18_quantized.pt")
FEATURE_MODEL_OUT = os.path.join(BASE_DIR, "resnet18_feature.pt")
LABELS_OUT = os.path.join(BASE_DIR, "imagenet_labels.json")
CLASS_INDEX_LOCAL = os.path.join(BASE_DIR, "imagenet_class_index.json")


def download_class_index():
    """下载 imagenet_class_index.json 到本地。"""
    if os.path.exists(CLASS_INDEX_LOCAL):
        print(f"[跳过] 已存在 {CLASS_INDEX_LOCAL}")
        return CLASS_INDEX_LOCAL

    last_err = None
    for url in CLASS_INDEX_URLS:
        try:
            print(f"[下载] 尝试从 {url} 下载类别索引 ...")
            urllib.request.urlretrieve(url, CLASS_INDEX_LOCAL)
            # 校验是否为合法 JSON
            with open(CLASS_INDEX_LOCAL, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == 1000, f"类别数量异常: {len(data)}"
            print(f"[成功] 已保存到 {CLASS_INDEX_LOCAL}")
            return CLASS_INDEX_LOCAL
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[失败] {url} -> {e}")

    raise RuntimeError(
        "无法下载 imagenet_class_index.json，请手动下载并放在项目根目录。\n"
        "参考源见 CLASS_INDEX_URLS。"
    ) from last_err


def build_label_mapping(class_index_path):
    """构建 {索引(int): 类别名(str)} 映射并保存为 imagenet_labels.json。"""
    with open(class_index_path, "r", encoding="utf-8") as f:
        class_idx = json.load(f)

    # 原始格式 {"0": ["n01440764", "tench"], ...} -> {0: "tench", ...}
    idx_to_label = {int(k): v[1] for k, v in class_idx.items()}

    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)

    print(f"[完成] 类别标签映射已保存到 {LABELS_OUT}（共 {len(idx_to_label)} 类）")
    return idx_to_label


def prepare_model():
    """下载预训练 ResNet-18，量化并保存为 TorchScript。"""
    print("[模型] 加载 torchvision 预训练 ResNet-18 ...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # ---- 动态量化：仅量化 Linear 层，减少内存、提升 CPU 推理速度 ----
    print("[模型] 执行动态量化 (qint8) ...")
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    model_quantized.eval()

    # ---- TorchScript 编译，消除 Python 解释开销，便于独立部署 ----
    print("[模型] 转换为 TorchScript ...")
    with torch.no_grad():
        scripted_model = torch.jit.script(model_quantized)

    # 预热一次前向传播，确保导出正常
    with torch.no_grad():
        scripted_model(torch.randn(1, 3, 224, 224))

    scripted_model.save(MODEL_OUT)
    print(f"[模型] 已保存量化模型到 {MODEL_OUT}")

    # 体积对比
    orig_size = sum(p.nelement() for p in model.parameters())
    print(f"[模型] 原始参数量: {orig_size/1e6:.2f}M")
    print(f"[模型] 模型文件大小: {os.path.getsize(MODEL_OUT)/1024/1024:.2f}MB")


def prepare_feature_model():
    """创建特征提取模型：去掉 ResNet-18 的最后全连接层，输出 512 维特征向量。"""
    print("[特征模型] 加载预训练 ResNet-18（去FC层）...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    # 用 Identity 替换最后的全连接层，前向传播输出 avgpool 后的 512 维向量
    model.fc = nn.Identity()

    with torch.no_grad():
        scripted = torch.jit.script(model)
        # 验证输出维度
        out = scripted(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 512), f"特征维度异常: {out.shape}"

    scripted.save(FEATURE_MODEL_OUT)
    print(f"[特征模型] 已保存到 {FEATURE_MODEL_OUT}（输出 512 维特征）")
    print(f"[特征模型] 文件大小: {os.path.getsize(FEATURE_MODEL_OUT)/1024/1024:.2f}MB")


def main():
    print("=" * 60)
    print("  ResNet-18 模型准备 (分类模型 + 特征提取模型)")
    print("=" * 60)

    # 1. 类别索引
    class_index_path = download_class_index()
    build_label_mapping(class_index_path)

    # 2. 分类模型量化与导出
    prepare_model()

    # 3. 特征提取模型（去FC层，输出512维）
    prepare_feature_model()

    print("=" * 60)
    print("全部完成！")
    print("  分类服务: python app.py")
    print("  物体识别Demo: python feature_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
