#!/bin/bash

# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    export TRANSFORMERS_CACHE=${work_dir}/.cache/huggingface
    export HF_MODULES_CACHE=${work_dir}/.cache/huggingface
# 02
else
    work_dir=/home/jinrenren
fi


model_size=${1}
num_bits=${2}
wait_pid=${3:--1}
max_new_tokens=${4}


if [[ ${#max_new_tokens} -eq 0 ]]; then
    args=""
else
    args="--max_new_tokens ${max_new_tokens}"
fi

dataset_name=snli
split=test

source ${work_dir}/miniconda3/etc/profile.d/conda.sh
checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
eval_data_path=${work_dir}/datasets/${dataset_name}
save_result_path=${work_dir}/quant_eval_results/${dataset_name}/${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat


conda activate quant

mkdir -p logs

nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} ${args} \
    --pid ${wait_pid} > logs/evaluate_spqr_quant_qwen_chat.${dataset_name}.${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
echo $!
