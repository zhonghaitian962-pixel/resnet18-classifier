FROM python:3.9-slim

# 避免 Python 写入 .pyc / 缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 先安装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 构建阶段：准备量化模型（需要网络下载预训练权重）
RUN python prepare_model.py

EXPOSE 8080

# 生产环境用 gunicorn，本地开发可用 python app.py
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
