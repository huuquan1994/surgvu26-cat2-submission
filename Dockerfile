FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir \
        transformers==4.53.3 \
        decord==0.6.0 \
        accelerate==1.10.0 \
        bitsandbytes==0.47.0 \
        opencv-python-headless==4.12.0.88 \
        albumentations==2.0.4 \
        numpy==2.2.6 \
        pillow==11.3.0 \
        PyYAML==6.0.2 \
    && mkdir -p /workspace /output

RUN groupadd -r user \
    && useradd -m --no-log-init -r -g user user \
    && chown -R user:user /workspace \
    && chmod -R 777 /workspace /output

COPY --chown=user:user task2_runtime /workspace/task2_runtime
COPY --chown=user:user vendor /workspace/vendor
COPY --chown=user:user models /workspace/models

RUN test -d /workspace/models/llava-onevision-qwen2-7b-ov-hf \
    && test -f /workspace/models/tools_classifier/efficientnet_v2_s_smoothing_0.025/epoch=11_test_loss=0.12283_f1_avg=0.97795.ckpt \
    && test -f /workspace/models/organs_classifier/efficientnet_v2_s_v3/epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt \
    && test -f /workspace/models/ec960_veto/config.yml \
    && test -f /workspace/models/ec960_veto/fold_0.pth \
    && test -f /workspace/models/ec960_veto/fold_1.pth \
    && test -f /workspace/models/ec960_veto/fold_2.pth \
    && test -f /workspace/models/ec960_veto/fold_3.pth \
    && test -f /workspace/models/ec960_veto/fold_4.pth

USER user
WORKDIR /workspace/task2_runtime
ENTRYPOINT ["python", "inference.py"]
