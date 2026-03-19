FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y ffmpeg curl git && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git /facefusion

WORKDIR /facefusion

RUN python install.py --onnxruntime cuda --skip-conda

RUN pip install runpod --no-cache-dir

RUN mkdir -p .assets/models && \
    curl -L -o .assets/models/ghost_1_256.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.hash && \
    curl -L -o .assets/models/ghost_1_256.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.onnx && \
    curl -L -o .assets/models/crossface_ghost.hash https://huggingface.co/facefusion/models-3.4.0/resolve/main/crossface_ghost.hash && \
    curl -L -o .assets/models/crossface_ghost.onnx https://huggingface.co/facefusion/models-3.4.0/resolve/main/crossface_ghost.onnx && \
    curl -L -o .assets/models/gfpgan_1.4.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.hash && \
    curl -L -o .assets/models/gfpgan_1.4.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx && \
    curl -L -o .assets/models/retinaface_10g.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.hash && \
    curl -L -o .assets/models/retinaface_10g.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx && \
    curl -L -o .assets/models/2dfan4.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.hash && \
    curl -L -o .assets/models/2dfan4.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx && \
    curl -L -o .assets/models/fan_68_5.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/fan_68_5.hash && \
    curl -L -o .assets/models/fan_68_5.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/fan_68_5.onnx && \
    curl -L -o .assets/models/arcface_w600k_r50.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.hash && \
    curl -L -o .assets/models/arcface_w600k_r50.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx && \
    curl -L -o .assets/models/fairface.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.hash && \
    curl -L -o .assets/models/fairface.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.onnx && \
    curl -L -o .assets/models/xseg_1.hash https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.hash && \
    curl -L -o .assets/models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx && \
    curl -L -o .assets/models/bisenet_resnet_34.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.hash && \
    curl -L -o .assets/models/bisenet_resnet_34.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx && \
    curl -L -o .assets/models/nsfw_1.hash https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_1.hash && \
    curl -L -o .assets/models/nsfw_1.onnx https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_1.onnx && \
    curl -L -o .assets/models/nsfw_2.hash https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_2.hash && \
    curl -L -o .assets/models/nsfw_2.onnx https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_2.onnx && \
    curl -L -o .assets/models/nsfw_3.hash https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_3.hash && \
    curl -L -o .assets/models/nsfw_3.onnx https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_3.onnx && \
    curl -L -o .assets/models/kim_vocal_2.hash https://huggingface.co/facefusion/models-3.0.0/resolve/main/kim_vocal_2.hash && \
    curl -L -o .assets/models/kim_vocal_2.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/kim_vocal_2.onnx

RUN mkdir -p /tmp/facefusion /tmp/facefusion-jobs

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
