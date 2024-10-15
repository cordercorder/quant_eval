#!/bin/bash

model_size=${1}
num_bits=${2}
export CUDA_VISIBLE_DEVICES=${3}
max_new_tokens=${4}

if [[ ${#max_new_tokens} -eq 0 ]]; then
    args=""
else
    args="--max_new_tokens ${max_new_tokens}"
fi

dataset_name=snli
split=test

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/work/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/${dataset_name}
    save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/${dataset_name}
    save_result_path=/home/jinrenren/quant_eval_results/${dataset_name}/${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
fi

conda activate quant

mkdir -p logs

nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} ${args} > logs/evaluate_gptq_quant_qwen_chat.${dataset_name}.${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1 &
