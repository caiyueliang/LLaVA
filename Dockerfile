FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG workdir=/workspace/taichu/
RUN mkdir -p ${workdir}
WORKDIR ${workdir}

COPY taichu/requirements.txt requirements.txt
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN apt-get update && apt-get install -y net-tools
COPY . LLaVA
RUN cd LLaVA && \
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN pip uninstall -y transformer-engine

# COPY vllm/taichu/entrypoint.sh entrypoint.sh
# RUN chmod +x entrypoint.sh

CMD ["bash", "./LLaVA/taichu/entrypoint.sh"]
