#!/bin/bash

model_size=${1}
num_bits=${2}
wait_pid=${3:--1}
dataset_name=bbq

# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
    eval_data_dir=${work_dir}/datasets/${dataset_name}
    source ${work_dir}/miniconda3/etc/profile.d/conda.sh
# 02
else
    work_dir=/home/jinrenren/jinrenren-starfs-data
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
    eval_data_dir=/home/jinrenren/datasets/${dataset_name}

    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

export TRANSFORMERS_CACHE=${work_dir}/.cache/huggingface
export HF_MODULES_CACHE=${work_dir}/.cache/huggingface


conda activate quant

save_result_dir=${work_dir}/quant_eval_results/${dataset_name}/spqr_${num_bits}g16_Qwen-${model_size}-Chat

mkdir -p logs

nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_dir ${eval_data_dir} \
    --save_result_dir ${save_result_dir} \
    --pid ${wait_pid} > logs/evaluate_spqr_quant_qwen_chat.${dataset_name}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
echo $!
