#!/bin/bash

HOST=${HOST:=0.0.0.0}
PORT=${PORT:=8080}
CONTROLLER_PORT=${CONTROLLER_PORT:=10000}
WEB_PORT=${WEB_PORT:=8081}
MODEL_PATH=${MODEL_PATH:=/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/}

# TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:=2}
# SEED=${SEED:=24}
# SWAP_SPACE=${SWAP_SPACE:=4}
# GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:=0.9}

# cmd="python -m vllm.entrypoints.openai.api_server \
# --host ${HOST} \
# --port ${PORT} \
# --model ${MODEL_PATH} \
# --trust-remote-code \
# --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
# --seed ${SEED} \
# --swap-space ${SWAP_SPACE} \
# --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
# "

# if [ -v MAX_MODEL_LEN ]; then
#     cmd+=" --max-model-len ${MAX_MODEL_LEN}"
# fi

# if [ -v MAX_PARALLEL_LOADING_WORKERS ]; then
#     cmd+=" --max-parallel-loading-workers ${MAX_PARALLEL_LOADING_WORKERS}"
# fi

# # export TAICHU_PREFIX=$'###答案：接下来是一个给我的提问或指令，我会解答这个问题并按照指令要求进行回复。</s>\n\n'

# if [ -v TAICHU_CHAT_TEMPLATE ]; then
#     cmd+=" --chat-template ${TAICHU_CHAT_TEMPLATE}"
# else
#     if [ -e "${MODEL_PATH}/template.jinja" ]; then
#         cmd+=" --chat-template ${MODEL_PATH}/template.jinja"
#     fi
# fi

# echo $cmd

# $cmd
cd /workspace/taichu/

# 启动 controller
nohup python -m llava.serve.controller --host ${HOST} --port ${CONTROLLER_PORT}  > controller.log 2>&1 &
while true; do
  # 使用netstat检测端口是否开启
  netstat -an | grep LISTEN | grep ":${CONTROLLER_PORT}"
  # 检查命令的返回状态码
  if [ $? -eq 0 ]; then
    echo "端口 ${CONTROLLER_PORT} 已经开启"
    break
  else
    echo "waiting port ${CONTROLLER_PORT}"
    sleep 2
  fi
done

# 启动 gradio_web_server (非必要)
nohup python -m llava.serve.gradio_web_server --controller http://localhost:${CONTROLLER_PORT} --port ${WEB_PORT} --model-list-mode reload > gradio_web_server.log 2>&1 &
while true; do
  # 使用netstat检测端口是否开启
  netstat -an | grep LISTEN | grep ":${WEB_PORT}"
  # 检查命令的返回状态码
  if [ $? -eq 0 ]; then
    echo "端口 ${WEB_PORT} 已经开启"
    break
  else
    echo "waiting port ${WEB_PORT}"
    sleep 2
  fi
done

# 启动 model_worker
python -m llava.serve.model_worker --host ${HOST} --controller http://localhost:10000 --port ${PORT} --worker http://localhost:${PORT} \
    --model-path ${MODEL_PATH}