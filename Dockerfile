#base image for jetpack 5

FROM nvcr.io/nvidia/l4t-base:r35.1.0

ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

#install system package, pip and opencv

RUN apt-get update && apt-get install -y \
    python3-pip \
    git\
    libopencv-dev \
    cuda-toolkit-11-4 \
    libcudnn8 \
    libnvinfer8 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel    
RUN pip3 install polars --only-binary=polars --index-url=https://pypi.org/simple
RUN pip3 install https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl 
# --- Build torchvision from source (JetPack fix for missing arch flags)
# --- Build torchvision from source (JetPack 5 compatible) ---
RUN apt-get update && apt-get install -y --no-install-recommends ninja-build && \
    git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    git clone --depth 1 --branch v0.15.1 https://github.com/pytorch/vision.git /tmp/vision && \
    cd /tmp/vision && \
    export BUILD_VERSION=0.15.1 && \
    export FORCE_CUDA=1 && \
    export CUDA_HOME=/usr/local/cuda && \
    export TORCH_CUDA_ARCH_LIST="7.2" && \
    python3 setup.py clean && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl && \
    rm -rf /tmp/vision


#copy files to the container
COPY ./requirement.txt /workspace/fishsize/

#install python packages
RUN pip3 install --no-cache-dir -r /workspace/fishsize/requirement.txt

#set the working directory
WORKDIR /workspace/fish_size


    

