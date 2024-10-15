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

dataset_name=alpaca_eval_eval

split=test

mkdir -p logs

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
eval_data_path=${work_dir}/datasets/tatsu-lab/alpaca_eval/alpaca_eval_eval
save_result_path=${work_dir}/quant_eval_results/${dataset_name}/${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat

CUDA_VISIBLE_DEVICES=${3} nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} > logs/evaluate_spqr_quant_qwen_chat.${dataset_name}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
