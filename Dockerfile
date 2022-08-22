FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04 AS base
LABEL maintaner=otavio.b.gomes@gmail.com

## TODO: delete this
RUN apt update && DEBIAN_FRONTEND=noninteractive apt -yq install aria2

WORKDIR /
RUN aria2c https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /miniconda3 && rm Miniconda3-py38_4.10.3-Linux-x86_64.sh
ENV PATH="/miniconda3/bin:${PATH}"

RUN conda install cudatoolkit=11.1 -c pytorch -c conda-forge
RUN conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge

ADD environment.yml .
RUN conda env update -n base --file environment.yml

###
# Devcontainer
FROM base AS devcontainer
LABEL maintaner=otavio.b.gomes@gmail.com

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -yq install sudo git byobu emacs\
	clang-format gdb cmake-qt-gui irony-server # nsight-compute

RUN chmod 777 /miniconda3/bin /miniconda3/lib/python3.8/site-packages

ADD requirements-dev.txt .
RUN pip install -U pip && pip install -r requirements-dev.txt

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

SHELL ["/bin/bash", "-c"]
