#!/bin/bash

model_size=${1}
torch_dtype=${2}

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
fi

conda activate quant

cd ../../

mkdir -p code/scripts/logs

CUDA_VISIBLE_DEVICES=${3} nohup python -u code/qwen_chat_inference.py \
    --model_name ${torch_dtype}.Qwen-${model_size}-Chat \
    --checkpoint_path ${checkpoint_path} \
    --torch_dtype ${torch_dtype} > code/scripts/logs/qwen_chat_inference.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
