#!/bin/bash

# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/huggingface
    export HF_MODULES_CACHE=/home/work/jinrenren/.cache/huggingface
# 02
else
    work_dir=/home/jinrenren
fi

source ${work_dir}/miniconda3/etc/profile.d/conda.sh

conda activate quant

# gpu=$1
model_size=$1
num_bits=$2
pid=${3:--1}

nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${work_dir}/pretrained_models/Qwen-${model_size}-Chat \
    --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat \
    --eval_data_path ${work_dir}/datasets/flores200_dataset/devtest/zho_Hans.devtest \
    --save_result_path ${work_dir}/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.spqr_${num_bits}g16_Qwen-${model_size}-Chat.devtest \
    --source_lang Chinese \
    --target_lang English \
    --pid ${4:--1} > evaluate_spqr_quant_qwen_chat.zh2en.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
echo pid $!
