FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

ARG git_owner
ARG git_repo
ARG git_branch
ARG snetd_version

ENV SINGNET_REPOS=/opt/singnet
ENV PROJECT_ROOT=${SINGNET_REPOS}/${git_repo}
ENV SERVICE_DIR=${PROJECT_ROOT}/service

ENV MODEL_PATH=${SERVICE_DIR}/models/segnet_final_reference.pth

# Updates and basic dependencies
RUN apt update &&\
    apt install wget git &&\
    cd ~ &&\
    python3 -m pip install cython &&\
    python3 -m pip install --upgrade pip

# Installing snet-daemon + dependencies
RUN mkdir snet-daemon && \
    cd snet-daemon && \
    wget https://github.com/singnet/snet-daemon/releases/download/${snetd_version}/snet-daemon-${snetd_version}-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-${snetd_version}-linux-amd64.tar.gz && \
    mv snet-daemon-${snetd_version}-linux-amd64/snetd /usr/bin/snetd && \
    cd .. && \
    rm -rf snet-daemon

# Cloning service repository and downloading models. Installing projects's original dependencies and building protobuf messages
RUN mkdir -p ${SINGNET_REPOS} &&\
    cd ${SINGNET_REPOS} &&\
    git clone -b ${git_branch} https://github.com/${git_owner}/${git_repo}.git &&\
    cd ${PROJECT_ROOT} &&\
    python3 -m pip install -r requirements.txt &&\
    sh buildproto.sh &&\
    ./service/download_models.py --filepath ${MODEL_PATH} &&\
    python3 -m pip uninstall -y matplotlib

WORKDIR ${PROJECT_ROOT}
