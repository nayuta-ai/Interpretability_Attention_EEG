FROM pytorch/pytorch:latest

RUN apt-get update && \
    apt-get install -y \
        git \
        wget \
        curl \
        tmux \
        sudo
WORKDIR /workspace/docker
COPY docker/requirements.txt /workspace/docker/
RUN pip install -r requirements.txt
WORKDIR /workspace