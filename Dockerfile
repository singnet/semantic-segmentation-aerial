FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04

ARG git_owner
ARG git_repo
ARG git_branch
ENV SINGNET_REPOS=/opt/singnet
ENV PROJECT_ROOT=${SINGNET_REPOS}/${git_repo}
ENV SERVICE_DIR=${PROJECT_ROOT}/service

# Installing requirements
RUN apt update && \
    apt install -y wget git build-essential software-properties-common

# Install Python3.6
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.6 python3-pip && \
    python3.6 -m pip install --upgrade pip

# Installing snet-daemon + dependencies
RUN mkdir snet-daemon && \
    cd snet-daemon && \
    wget -q https://github.com/singnet/snet-daemon/releases/download/v0.1.8/snet-daemon-v0.1.8-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-v0.1.8-linux-amd64.tar.gz  && \
    mv ./snet-daemon-v0.1.8-linux-amd64/snetd /usr/bin/snetd && \
    cd .. && \
    rm -rf snet-daemon

# Cloning service repository and downloading models
RUN mkdir -p ${SINGNET_REPOS} && \
    cd ${SINGNET_REPOS} &&\
    git clone -b ${git_branch} https://github.com/${git_owner}/${git_repo}.git
    cd ${SERVICE_DIR} &&\
    . ./service/download_models.py

# Installing projects's original dependencies and building protobuf messages
RUN cd ${PROJECT_ROOT} &&\
    python3.6 -m pip install -r requirements.txt &&\
    sh buildproto.sh

WORKDIR ${PROJECT_ROOT}
