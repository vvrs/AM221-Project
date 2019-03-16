# Dockerfile containing required software for this project

FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER Vishnu Rudrasamudram (vishnu.rudrasamudram@gmail.com)

# set home location
ENV HOME=/home
ENV TERM=xterm-256color

RUN apt-get update && \
    apt-get install -y curl git zsh vim wget zip

# install and add conda to path
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh && \
    /bin/bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p /opt/conda

ENV PATH /opt/conda/bin:$PATH

RUN conda update -n base conda && \
    conda install pytorch=0.4 torchvision -c pytorch -y

RUN pip install --upgrade pip && \
    pip install setproctitle line_profiler setGPU waitGPU dotfiles

RUN echo cd >> $HOME/.bashrc

ENV PYTHONPATH /home/:$PATH
ENV CUDA_DEVICE_ORDER PCI_BUS_ID
ENV LANG C.UTF-8 
ENV LC_ALL C.UTF-8

# MNIST data
COPY . /home/
#COPY convex_adversarial /home/convex_adversarial/
#COPY examples /home/examples/
