#!/bin/bash

source /home/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

export TRANSFORMERS_CACHE=/home/jinrenren/.cache/transformers
export HF_MODULES_CACHE=/home/jinrenren/.cache/transformers/huggingface

split=${1}
model_size=${2}
num_bits=${3}

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
eval_data_path=/home/jinrenren/datasets/ceval
save_result_dir=/home/jinrenren/quant_eval_results/ceval.${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat

CUDA_VISIBLE_DEVICES=${4} nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_dir ${save_result_dir} \
    --split ${split} > evaluate_gptq_quant_qwen_chat.${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1 &
