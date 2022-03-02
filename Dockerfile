FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

RUN python3 -m pip install segmentation-models \
pip install scikit-learn \
pip install opencv-python pandas

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow

# permissions to user
RUN chmod 777 /home/tensorflow && \
chown ${uid}:${gid} -R /home/tensorflow
ENV HOME /home/tensorflow

WORKDIR /home/tensorflow

ENV TF_CPP_MIN_LOG_LEVEL 3