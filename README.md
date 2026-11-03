# ResNet18 物体识别 Demo（特征库 + 语音交互）

基于预训练 ResNet18 的本地物体识别演示。既可以做标准 ImageNet 1000 类分类，也可以自己录入物体、建立专属特征库，并通过「上传图片 / 摄像头 / 语音报名称」完成识别。

- 特征提取：ResNet18 去掉全连接层，输出 512 维向量，余弦相似度检索
- 特征库：一个物体可录入多张图片，可单独删除任意一张图片或整个词条
- 未知物体：相似度低于阈值时回退到 ImageNet 分类，给出可能类别
- 语音交互：浏览器 Web Speech API，语音输入物体名称 + 语音播报结果（需 Chrome / Edge）
- 本地持久化：特征库存入 `feature_db.pkl`，图片存入 `feature_images/`

## 环境要求

- Python 3.9 ~ 3.11
- 约 200MB 磁盘空间（含 PyTorch 与模型权重）
- 首次运行需要联网下载预训练 ResNet18 权重（约 45MB）
- 建议把项目放在**纯英文路径**下，避免 TorchScript 在中文路径下加载模型失败

## 快速开始

```bash
# 1. 创建并激活虚拟环境（Windows PowerShell 示例）
python -m venv venv
venv\Scripts\activate

# macOS / Linux
# python3 -m venv venv && source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 生成模型文件（只需执行一次）
#    产出 resnet18_quantized.pt / resnet18_feature.pt / imagenet_labels.json
python prepare_model.py
```

> 仓库不包含模型权重和本地特征库，克隆后必须执行第 3 步。

### 启动方式一：ImageNet 分类 Demo

```bash
python app.py
```

打开 http://localhost:8080 ，上传一张图片即可看到 Top-3 类别（英文 + 中文）。

### 启动方式二：特征库 / 物体识别 Demo（主功能）

```bash
python feature_app.py
```

打开 http://localhost:8081/manage 。

请使用 `localhost` 访问，不要用局域网 IP：摄像头与语音识别在非安全上下文中会被浏览器禁用。

Windows 也可以双击 `run.bat`：脚本会自动定位同目录的 `venv` 或系统 Python，缺依赖时自动安装，首次运行自动生成模型，然后启动服务并打开浏览器。

## 使用流程（特征库 Demo）

1. **录入物体**：切换到「录入物体」唱片，输入名称，上传或拍摄多张图片，提交后自动提取特征并入库。
2. **识别物体**：切换到「识别物体」唱片，上传图片或调用摄像头；说了物体名称也可以直接语音检索。
3. **管理特征库**：切换到「特征库」唱片
   - 查看每个词条的全部图片（图片多时区域内滚动，不撑高表格）
   - 点缩略图右上角红色 × 删除**任意一张**图片及其对应特征
   - 删除整个词条，或清空特征库
   - 「为词条补充图片」可继续给已有物体添加样本

## 关键参数与数据

| 项目 | 说明 |
| --- | --- |
| 相似度阈值 | `feature_app.py` 中 `SIMILARITY_THRESHOLD = 0.75`，低于则判为未知并回退 ImageNet |
| 特征维度 | 512 |
| 特征库文件 | `feature_db.pkl`（pickle，本地自动生成） |
| 图片目录 | `feature_images/<物体名>/<编号>.jpg` |
| 备份 / 迁移 | 同时拷贝 `feature_db.pkl` 和 `feature_images/` 即可 |
| 重置特征库 | 页面里点「清空全部特征库」，或删除上述文件与目录 |

## 把项目发给朋友

**推荐：直接发 GitHub 仓库或 ZIP 下载链接**（代码随仓库更新，不用来回传压缩包）：

- 仓库：https://github.com/zhonghaitian962-pixel/resnet18-classifier
- ZIP：https://github.com/zhonghaitian962-pixel/resnet18-classifier/archive/refs/heads/main.zip

如果一定要发本地压缩包，请注意：

1. **压缩整个项目文件夹**，不要只发 `run.bat`（脚本需要同目录的 `feature_app.py`、`templates/` 等）。
2. 对方**先完整解压**，再进入文件夹双击 `run.bat`；不要直接在压缩包预览窗口里运行。
3. 对方电脑需要安装 **Python 3.9 ~ 3.11**（安装时勾选“Add python.exe to PATH”），首次运行需要联网下载依赖和模型。
4. **不要打包 `venv/`**：虚拟环境与打包电脑绑定，换机可能失效，而且体积很大。`run.bat` 会自动用系统 Python 安装依赖。
5. 解压路径尽量用英文，中文路径可能导致 TorchScript 加载模型失败。
6. 仓库/压缩包不含 `*.pt` 模型和 `feature_db.pkl`，属于正常现象；`run.bat` 首次运行会自动执行 `prepare_model.py`。

## Docker（仅分类 Demo）

```bash
docker build -t resnet18-demo .
docker run -p 8080:8080 resnet18-demo
```

镜像构建时会自动执行 `prepare_model.py`（需要联网），容器内只启动 `app.py`（端口 8080）。特征库 Demo 请在本机运行，或自行修改 `Dockerfile` 的启动命令为 `python feature_app.py` 并开放 8081。

## 接口一览

分类服务（`app.py`，8080）：

- `GET /` 上传页；`POST /` 上传字段 `image`
- `GET /health` 健康检查

特征库服务（`feature_app.py`，8081）：

- `GET /manage` 管理页面
- `POST /register` 字段 `name` + 多张 `images`
- `POST /recognize` 字段 `image`
- `GET /database` 特征库摘要（含每张图片的真实索引）
- `POST /delete` 字段 `name`
- `POST /delete_image` 字段 `name` + `index`
- `POST /clear` 清空特征库
- `GET /health`

## 常见问题

- **启动时报找不到 `resnet18_feature.pt` / `resnet18_quantized.pt`**：先运行 `python prepare_model.py`。
- **首次 `prepare_model.py` 卡在下载**：需要访问 PyTorch 官方权重地址，换个网络或配置代理后重试。
- **语音按钮无反应**：确认使用 Chrome / Edge，并通过 `http://localhost:8081` 访问；Firefox 对中文语音识别支持有限。
- **摄像头打不开**：检查浏览器摄像头权限与是否被其他程序占用，同样必须使用 localhost。
- **`RuntimeError: open file failed ... errno 2`（模型明明存在）**：TorchScript 对中文路径兼容性差，把项目移动到英文路径（如 `C:\projects\resnet18-classifier`）后重试。
- **端口被占用**：`app.py` 用 8080、`feature_app.py` 用 8081，可在各自文件末尾的 `app.run(...)` 中修改。
- **识别总是不准**：每个物体建议录入 3~10 张不同角度/光照的图片；必要时降低或提高 `SIMILARITY_THRESHOLD`。

## 目录结构

```
resnet18-classifier/
├── app.py                 # ImageNet 分类 Demo（8080）
├── feature_app.py         # 特征库 / 物体识别 Demo（8081）
├── prepare_model.py       # 生成分类模型、特征模型、类别标签
├── templates/             # 页面模板（feature_index.html 为主要界面）
├── static/                # 静态资源
├── demos/                 # 早期布局草稿
├── requirements.txt
├── Dockerfile
└── run.bat                # Windows 一键启动（需修改路径）
```
