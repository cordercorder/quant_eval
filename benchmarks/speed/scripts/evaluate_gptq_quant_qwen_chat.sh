#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
num_bits=${2}
device=${3}

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

dataset_name=pg19
filed_name=text

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/work/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/pg19/test
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/pg19/test
fi

mkdir -p logs

batch_size=1
input_length=256
num_new_tokens=512

if [[ ${#device} -eq 0 ]]; then
    python -u ../evaluate_gptq_quant_qwen_chat.py \
        --checkpoint_path ${checkpoint_path} \
        --quant_checkpoint_path ${quant_checkpoint_path} \
        --eval_data_path ${eval_data_path} \
        --filed_name ${filed_name} \
        --batch_size ${batch_size} \
        --input_length ${input_length} \
        --num_new_tokens ${num_new_tokens} > logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1
else
    CUDA_VISIBLE_DEVICES=${device} python -u ../evaluate_gptq_quant_qwen_chat.py \
        --checkpoint_path ${checkpoint_path} \
        --quant_checkpoint_path ${quant_checkpoint_path} \
        --eval_data_path ${eval_data_path} \
        --filed_name ${filed_name} \
        --batch_size ${batch_size} \
        --input_length ${input_length} \
        --num_new_tokens ${num_new_tokens} > logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1
fi
