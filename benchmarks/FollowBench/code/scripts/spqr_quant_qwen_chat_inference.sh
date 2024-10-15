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

source ${work_dir}/miniconda3/etc/profile.d/conda.sh

conda activate quant


model_size=${1}
num_bits=${2}
# gpu=${3}
wait_pid=${3:--1}



checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
# spqr_w3g16_Qwen-14B-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat

cd ../../

mkdir -p code/scripts/logs

# example: 3张卡
nohup python -u code/spqr_quant_qwen_chat_inference.py \
    --model_name spqr_${num_bits}g16_Qwen-${model_size}-Chat \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --pid ${wait_pid} > code/scripts/logs/spqr_quant_qwen_chat_inference.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &

echo $!