ARG BASE_IMAGE=nvcr.io/nvidia/pytorch
ARG BASE_TAG=23.10-py3
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG workdir=/workspace/taichu/
RUN mkdir -p ${workdir}
WORKDIR ${workdir}

RUN pip install llava-torch==1.2.2.post1 -i https://pypi.douban.com/simple/
COPY . llava
RUN cd llava && \
    pip install -e . -i https://pypi.douban.com/simple/


# COPY vllm/taichu/entrypoint.sh entrypoint.sh
# RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash ./llava/taichu/entrypoint.sh"]
