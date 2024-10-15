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

if [[ ${#device} -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES=${device}
fi

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

dataset_name=mt_bench

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/work/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    question_file=/home/work/jinrenren/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
    answer_file=/home/work/jinrenren/quant_eval_results/${dataset_name}/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.jsonl
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    question_file=/home/jinrenren/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
    answer_file=/home/jinrenren/quant_eval_results/${dataset_name}/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.jsonl
fi

mkdir -p logs

nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --question_file ${question_file} \
    --answer_file ${answer_file} > logs/evaluate_gptq_quant_qwen_chat.${dataset_name}.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1 &
