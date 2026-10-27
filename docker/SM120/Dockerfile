# syntax=docker/dockerfile:1
# The digest pins the public vLLM base. Model weights are mounted separately.
FROM vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
USER root
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg git patch numactl libsndfile1 libgl1 \
    libibverbs-dev librdmacm-dev ibverbs-providers rdma-core \
    && rm -rf /var/lib/apt/lists/*
COPY docker/SM120/requirements.txt /opt/h3-requirements.txt
RUN python3 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    'torch==2.13.0+cu132' 'torchvision==0.28.0+cu132' \
    && python3 -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cpu 'torchaudio==2.11.0+cpu' \
    && python3 -m pip install --no-cache-dir -r /opt/h3-requirements.txt
# Match the separately recorded JIT toolchain, rather than the base image nvcc.
RUN python3 -m pip install --no-cache-dir --no-deps --target /opt/h3-cuda \
    nvidia-cuda-nvcc==13.2.78 nvidia-cuda-crt==13.2.78 nvidia-nvvm==13.2.78 \
    nvidia-cuda-runtime==13.2.75 nvidia-cuda-cccl==13.2.75 \
    nvidia-nvjitlink==13.2.78 nvidia-cublas==13.4.0.1 \
    nvidia-cusolver==12.2.0.1 nvidia-cusparse==12.7.10.1
WORKDIR /opt/h3
COPY . /opt/h3
RUN python3 scripts/prepare_flashinfer.py \
    && python3 scripts/sku_benchmark.py prepare --variant avatar5 --output /opt/h3-runtime-avatar5 \
    && python3 scripts/sku_benchmark.py prepare --variant reference --output /opt/h3-runtime-15s
ENV CUDA_HOME=/opt/h3-cuda/nvidia/cu13 \
    FLASHINFER_NVCC=/opt/h3-cuda/nvidia/cu13/bin/nvcc \
    PATH=/opt/h3-cuda/nvidia/cu13/bin:$PATH
RUN ln -s /usr/local/bin/vllm /usr/bin/vllm \
    && python3 docker/SM120/container.py check \
    && python3 -m pip freeze > /opt/h3-image-packages.txt
ENTRYPOINT ["python3", "/opt/h3/docker/SM120/container.py"]
CMD ["--help"]
