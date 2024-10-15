#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
torch_dtype=bf16

dataset_name=mt_bench

split=test

mkdir -p logs

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    question_file=/home/work/jinrenren/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
    answer_file=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.${torch_dtype}.Qwen-${model_size}-Chat.jsonl
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    question_file=/home/jinrenren/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
    answer_file=/home/jinrenren/quant_eval_results/${dataset_name}/${split}.${torch_dtype}.Qwen-${model_size}-Chat.jsonl
fi

CUDA_VISIBLE_DEVICES=${2} nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --question_file ${question_file} \
    --answer_file ${answer_file} > logs/evaluate_qwen_chat.${dataset_name}.${split}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
