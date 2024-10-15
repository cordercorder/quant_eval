#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
num_bits=${2}
dataset_name=${3}
export CUDA_VISIBLE_DEVICES=${4}
max_new_tokens=${5}

if [[ ${#max_new_tokens} -eq 0 ]]; then
    args=""
else
    args="--max_new_tokens ${max_new_tokens}"
fi

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"


if [[ ${dataset_name} = "cnn_dailymail" ]]; then
    document_key=article
    summary_key=highlights
else
    document_key=document
    summary_key=summary
fi

split=test

mkdir -p logs

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/work/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/${dataset_name}
    save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/${dataset_name}
    save_result_path=/home/jinrenren/quant_eval_results/${dataset_name}/${split}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
fi

nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} \
    --document_key ${document_key} \
    --summary_key ${summary_key} ${args} > logs/evaluate_gptq_quant_qwen_chat.${dataset_name}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1 &
