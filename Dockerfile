FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /training
COPY requirements.txt  /training
RUN pip install -r requirements.txt 