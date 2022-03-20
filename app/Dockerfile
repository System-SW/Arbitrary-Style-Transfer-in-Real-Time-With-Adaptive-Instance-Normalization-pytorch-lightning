FROM python:3.8.12-slim-buster
EXPOSE 8501
WORKDIR /app

RUN apt-get update 
RUN apt install -y \ 
    libgl1-mesa-glx \
    libglib2.0-0 && apt-get clean


ENV MODEL_PATH ./src/AdaIN.pt.zip

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]