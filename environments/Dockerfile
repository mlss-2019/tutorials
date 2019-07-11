FROM jupyter/base-notebook

MAINTAINER Isak Falk (https://isakfalk.com)
USER root
# Install necessary jdk8 stuff
RUN apt-get update --yes
RUN apt-get upgrade --yes
RUN apt-get install build-essential openjdk-8-jdk gcc git --yes --force-yes
RUN git clone https://github.com/mlss-2019/tutorials ./tutorials
WORKDIR "./tutorials/environments/"

RUN conda env create -f mlss_gp.yml
# RUN conda env create -f mlss_mcmc.yml
RUN conda env create -f mlss_rl.yml
RUN conda env create -f mlss_vi.yml
RUN conda install nb_conda

# Get Xserver to work (from https://github.com/manuelsh/minerl-docker/blob/master/Dockerfile)
# This if for minerl
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \
    xvfb \
    python-opengl\
    x11-xserver-utils\
&& rm -rf /var/lib/apt/lists/*
# Create starting file
RUN echo "xhost + & jupyter notebook --allow-root --no-browser" > /xvfb.sh
RUN echo "xvfb-run -s \"-screen 0 1400x900x24\" /xvfb.sh" > /run.sh
RUN chmod 777 /xvfb.sh
RUN chmod 777 /run.sh

WORKDIR ".."
ENTRYPOINT ["sh", "/run.sh"]