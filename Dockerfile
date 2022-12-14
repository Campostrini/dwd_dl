# Copyright 2022 Stefano Campostrini
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/cuda:11.0.3-base-ubuntu18.04
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN conda update conda

RUN conda config --set channel_priority flexible

COPY environment.yml .
RUN conda env create -f environment.yml

RUN echo "export PYTHONPATH=/home/dwd_dl/" >> /root/.bashrc

RUN conda init

RUN bash /root/.bashrc

EXPOSE 6006
EXPOSE 7777

SHELL ["conda", "run", "-n", "py38", "/bin/bash", "-c"]