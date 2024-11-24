FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils git wget curl unzip && \ 
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install kaggle

RUN kaggle datasets download -d vasiliygorelov/2k2k2k2k && \
    unzip 2k2k2k2k.zip -d /app/ && \
    rm 2k2k2k2k.zip
    
RUN kaggle datasets download -d vasiliygorelov/2k2k2kcsv && \
    unzip 2k2k2kcsv.zip -d /app/ && \
    rm 2k2k2kcsv.zip

FROM builder AS stage

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .
CMD ["python3.9", "train.py"]
