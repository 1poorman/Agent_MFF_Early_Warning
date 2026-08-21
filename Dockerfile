# ============================================================================
# 中频炉水冷系统多参数融合预警智能体 —— Docker 镜像
# ----------------------------------------------------------------------------
# 基于 conda 环境 mff_agent（python 3.10），与本地开发环境完全一致。
#
# 构建：
#   docker build -t mff-agent:latest .
#   # 如需 CUDA 版 PyTorch：docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t mff-agent:latest .
#
# 启动（推荐 docker compose）：
#   docker compose up -d
# ============================================================================

FROM continuumio/miniconda3:24.7.1-0

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    MFF_APP_ENV=docker

WORKDIR /app

# ---- 1. 创建 mff_agent conda 环境（python 3.10） ----
RUN conda create -n mff_agent python=3.10 -y && \
    conda clean -afy

# ---- 2. 安装依赖（利用 Docker 层缓存：依赖文件变更才重装） ----
COPY requirements.txt .
RUN /opt/conda/envs/mff_agent/bin/pip install --no-cache-dir -r requirements.txt

# ---- 3. 安装 PyTorch（默认 CPU 版保证通用；可传参切换 CUDA index） ----
ARG TORCH_VERSION=2.9.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN /opt/conda/envs/mff_agent/bin/pip install --no-cache-dir \
        torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}

# ---- 4. 复制项目代码 ----
COPY . .

# ---- 5. 默认使用 mff_agent 环境 ----
ENV PATH=/opt/conda/envs/mff_agent/bin:$PATH

# 模型权重 / 数据 / 日志通过 volume 挂载（不随镜像分发）
VOLUME ["/app/models", "/app/data", "/app/logs"]

EXPOSE 8000 8100

# 默认启动 RESTful API + Web 界面
CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
