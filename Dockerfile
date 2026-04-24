ARG PYTORCH="2.2.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=4
ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ninja-build cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Symlink libcudart so MinkowskiEngine can find it at compile time
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/lib/libcudart.so

WORKDIR /workspace

# Install torch_geometric and PyG sparse extensions
# (torch/torchvision/torchaudio are already provided by the base image)
RUN pip install --no-cache-dir torch_geometric && \
    pip install --no-cache-dir \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Clone patched MinkowskiEngine and build it
RUN git clone --depth 1 https://github.com/renezurbruegg/MinkowskiEngine.git /workspace/MinkowskiEngine
RUN cd /workspace/MinkowskiEngine && \
    python setup.py install \
        --force_cuda \
        --blas=openblas \
        --blas_include_dirs=/usr/include/openblas \
        --cuda_home=${CUDA_HOME}

# Copy the rest of the repo
COPY . /workspace/icg_net
WORKDIR /workspace/icg_net

# Build pointnet2 CUDA extensions
RUN cd icg_net/third_party/pointnet2 && python setup.py install

# Install training dependencies
RUN pip install --no-cache-dir \
    'numpy<2' \
    hydra-core \
    omegaconf \
    pytorch-lightning \
    wandb \
    albumentations \
    elasticdeform \
    python-dotenv \
    trimesh \
    torchmetrics \
    rich \
    matplotlib \
    loguru \
    scikit-image \
    scipy

# Install icg_net package in editable mode
RUN pip install --no-cache-dir -e .
