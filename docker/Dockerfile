# MindSpore 2.0.0 GPU training image for SageMaker
# CUDA 11.6 + cuDNN 8 + Python 3.9
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.9 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3-pip python3.9-distutils \
    libgomp1 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install MindSpore 2.0.0 + training deps
RUN pip install --no-cache-dir \
    mindspore==2.0.0 \
    numpy \
    omegaconf \
    scikit-learn \
    psutil \
    "Pillow<10" \
    protobuf

# SageMaker expects the training script at /opt/ml/code/
RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/input/data/training

COPY entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
