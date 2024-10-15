#!/bin/bash

model_size=${1}
num_bits=${2}

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/work/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
fi

conda activate quant

cd ../../

mkdir -p code/scripts/logs

CUDA_VISIBLE_DEVICES=${3} nohup python -u code/gptq_quant_qwen_chat_inference.py \
    --model_name gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} > code/scripts/logs/gptq_quant_qwen_chat_inference.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1 &
