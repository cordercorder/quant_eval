#!/bin/bash

source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

model_size=${1}
torch_dtype=${2}

dataset_name=gsm8k

split=test

checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
eval_data_path=/home/work/jinrenren/datasets/${dataset_name}
save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.${torch_dtype}.Qwen-${model_size}-Chat

mkdir -p logs

CUDA_VISIBLE_DEVICES=${3} nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} \
    --torch_dtype ${torch_dtype} > logs/evaluate_qwen_chat.${dataset_name}.${split}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
