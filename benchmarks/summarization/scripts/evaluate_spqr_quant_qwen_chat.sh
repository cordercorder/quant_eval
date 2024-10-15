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

export TRANSFORMERS_CACHE=${work_dir}/.cache/transformers
export HF_MODULES_CACHE=${work_dir}/.cache/transformers/huggingface
export TRITON_CACHE_DIR=${work_dir}/.cache
export TMPDIR=${work_dir}/tmp

model_size=${1}
num_bits=${2}

dataset_name=cnn_dailymail
document_key=article
summary_key=highlights

split=test

mkdir -p logs

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
eval_data_path=${work_dir}/datasets/${dataset_name}
save_result_path=${work_dir}/quant_eval_results/${dataset_name}/${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat

nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} \
    --document_key ${document_key} \
    --summary_key ${summary_key} \
    --pid -1 > logs/evaluate_spqr_quant_qwen_chat.${dataset_name}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
echo last pid $!