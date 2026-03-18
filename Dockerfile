FROM madiator2011/better-pytorch:cuda12.4-torch2.6.0

RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

RUN git clone --branch 3.4.1 --depth 1 https://github.com/facefusion/facefusion.git /facefusion

WORKDIR /facefusion

RUN pip install numpy==2.3.2 onnx==1.19.0 opencv-python==4.12.0.88 \
    psutil==7.0.0 tqdm==4.67.1 scipy==1.16.1 \
    runpod --no-cache-dir

RUN pip install onnxruntime-gpu==1.22.1 --no-cache-dir

RUN python facefusion.py force-download \
    --processors face_swapper face_enhancer \
    --face-swapper-model ghost_1_256 \
    --face-enhancer-model gfpgan_1.4 \
    --face-detector-model retinaface \
    --face-landmarker-model 2dfan4 \
    --execution-providers cuda

RUN mkdir -p /tmp/facefusion /tmp/facefusion-jobs

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
