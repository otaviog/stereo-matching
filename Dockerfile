FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04 AS base
LABEL maintaner=otavio.b.gomes@gmail.com

## TODO: delete this
RUN apt update && DEBIAN_FRONTEND=noninteractive apt -yq install aria2 build-essential git

WORKDIR /
RUN aria2c https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /miniconda3 && rm Miniconda3-py38_4.10.3-Linux-x86_64.sh
ENV PATH="/miniconda3/bin:${PATH}"

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN pip install pypfm
ADD environment.yml .
RUN conda env update -n base --file environment.yml

# Supress DWARF warnings messages.
RUN rm /miniconda3/compiler_compat/ld && ln -s /usr/bin/ld /miniconda3/compiler_compat/ld

###
# Devcontainer
FROM base AS devcontainer
LABEL maintaner=otavio.b.gomes@gmail.com

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -yq install sudo git byobu emacs\
	clang-format gdb cmake-qt-gui irony-server # nsight-compute

RUN chmod 777 /miniconda3/bin /miniconda3/lib/python3.8/site-packages

ADD requirements-dev.txt .
RUN pip install -U pip && pip install -r requirements-dev.txt

ADD requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
# stereo-mideval downgrades numpy, lets update it again.
RUN pip install -U numpy

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
