FROM jupyter/base-notebook

MAINTAINER Isak Falk (https://isakfalk.com)

# Change to root
USER root
# Install necessary jdk8 stuff
# install necessary xorg stuff for minerl, get Xserver to work (from https://github.com/manuelsh/minerl-docker/blob/master/Dockerfile)

RUN apt-get update && apt-get install -y —no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    xvfb \
    python-opengl\ 
    x11-xserver-utils \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/* && \
    echo "xhost + & jupyter notebook --allow-root --no-browser" > /xvfb.sh && \
    echo "xvfb-run -s \"-screen 0 1400x900x24\" /xvfb.sh" > /run.sh  && \
    chmod 777 /xvfb.sh && chmod 777 /run.sh && \
    apt-get autoremove && apt-get clean

RUN git clone https://github.com/isakfalk/mlss-2019/tutorials ./tutorials
WORKDIR "./tutorials/environments/"

RUN conda env create -f mlss.yml && conda install nb_conda

# Install julia
RUN /opt/conda/envs/mlss/bin/julia -e 'using Pkg; ENV["python"] = "/opt/conda/envs/mlss/bin/python"; Pkg.add("Distributions"); Pkg.add("SpecialFunctions"); Pkg.add("PyPlot"); Pkg.add("IJulia")' \
    && /opt/conda/envs/mlss/bin/julia -e "using Distributions; using SpecialFunctions; using PyPlot'

WORKDIR ".."
ENTRYPOINT ["sh", "/run.sh"]