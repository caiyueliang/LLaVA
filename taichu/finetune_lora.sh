#!/bin/bash

# 初始化变量
pretrained_model_path="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/"
vision_tower="/mnt/publish-data/pretrain_models/llava/clip-vit-large-patch14-336/"
output_path="./output/"

# 遍历所有传递给脚本的参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --pretrained_model_path)
      pretrained_model_path="$2" # 获取 --pretrained_model_path 后面的参数值
      shift # 移过选项名
      shift # 移过选项值
      ;;
    --vision_tower)
      vision_tower="$2"
      shift
      shift
      ;;
    --output_path)
      output_path="$2"
      shift
      shift
      ;;

    *)    # 对于不认识的选项，直接报错或者忽略
      echo "unknow params: $1"
      exit 1
      ;;
  esac
done

# 使用变量
echo "======================================================================"
echo "[pretrained_model_path]: ${pretrained_model_path}"
echo "[vision_tower]: ${vision_tower}"
echo "[output_path]: ${output_path}"
echo "======================================================================"

# HOST=${HOST:=0.0.0.0}
# PORT=${PORT:=8080}
# CONTROLLER_PORT=${CONTROLLER_PORT:=10000}
# WEB_PORT=${WEB_PORT:=8081}
# MODEL_PATH=${MODEL_PATH:=/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/}

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --pretrained_model_path ${pretrained_model_path} \
    --vision_tower ${vision_tower} \
    --output_path ${output_path} \
    --data_path "/mnt/publish-data/train_data/llava_data/01/" \
    --version v1 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
