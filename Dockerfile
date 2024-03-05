FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG workdir=/workspace/taichu/
RUN mkdir -p ${workdir}
WORKDIR ${workdir}

COPY taichu/requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

COPY . llava
RUN cd llava && \
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/


# COPY vllm/taichu/entrypoint.sh entrypoint.sh
# RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash ./llava/taichu/entrypoint.sh"]
