#!/bin/bash

# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
# 02
else
    work_dir=/home/jinrenren
fi

source ${work_dir}/miniconda3/etc/profile.d/conda.sh

conda activate quant

model_size=${1}
num_bits=${2}

dataset_name=mt_bench

split=test

mkdir -p logs

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
question_file=${work_dir}/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
answer_file=${work_dir}/quant_eval_results/${dataset_name}/${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.jsonl

CUDA_VISIBLE_DEVICES=${3} nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --question_file ${question_file} \
    --answer_file ${answer_file} > logs/evaluate_spqr_quant_qwen_chat.${dataset_name}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
