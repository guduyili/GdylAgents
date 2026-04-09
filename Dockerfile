FROM python:3.11-slim

WORKDIR /app

# 切换 apt 为阿里云镜像
RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    (echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
     echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
     echo "deb https://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list)

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（用阿里云 PyPI 镜像加速）
RUN pip install uv --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/

# 配置 uv 使用阿里云 PyPI 镜像
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

# 先复制依赖文件，利用 Docker 层缓存
COPY pyproject.toml uv.lock ./

# 安装项目依赖（不安装项目本身）
RUN uv sync --frozen --no-install-project

# 复制项目代码
COPY . .

# 安装项目本身
RUN uv sync --frozen

# 暴露 LangGraph API 默认端口
EXPOSE 2024

# 数据持久化目录
VOLUME ["/data"]

# 启动 langgraph dev（--db-file 持久化到 SQLite，容器重启数据不丢失）
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser"]
