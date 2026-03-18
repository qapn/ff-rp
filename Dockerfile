FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y ffmpeg curl git && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git /facefusion

WORKDIR /facefusion

RUN python install.py --onnxruntime cuda --skip-conda

RUN pip install runpod --no-cache-dir

RUN python facefusion.py force-download

RUN mkdir -p /tmp/facefusion /tmp/facefusion-jobs

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
