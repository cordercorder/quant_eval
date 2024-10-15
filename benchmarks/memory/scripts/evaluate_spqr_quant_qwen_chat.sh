#!/bin/bash
# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
else
    work_dir=/home/jinrenren
fi

source ${work_dir}/miniconda3/etc/profile.d/conda.sh

conda activate quant

model_size=${1}
num_bits=${2}
device=${3}

dataset_name=pg19
filed_name=text

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
# spqr_w3g16_Qwen-14B-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
eval_data_path=${work_dir}/datasets/pg19/test
save_result_path=${work_dir}/quant_eval_results/memory/${dataset_name}/spqr_${num_bits}g16_Qwen-${model_size}-Chat

batch_size=1
input_length=256
num_new_tokens=512

if [[ ${#device} -eq 0 ]]; then
    python -u ../evaluate_spqr_quant_qwen_chat.py \
        --checkpoint_path ${checkpoint_path} \
         --quant_checkpoint_path ${quant_checkpoint_path} \
        --eval_data_path ${eval_data_path} \
        --filed_name ${filed_name} \
        --batch_size ${batch_size} \
        --input_length ${input_length} \
        --num_new_tokens ${num_new_tokens} \
        --save_result_path ${save_result_path} > logs/evaluate_spqr_quant_qwen_chat.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1
else
    CUDA_VISIBLE_DEVICES=${device} python -u ../evaluate_spqr_quant_qwen_chat.py \
        --checkpoint_path ${checkpoint_path} \
         --quant_checkpoint_path ${quant_checkpoint_path} \
        --eval_data_path ${eval_data_path} \
        --filed_name ${filed_name} \
        --batch_size ${batch_size} \
        --input_length ${input_length} \
        --num_new_tokens ${num_new_tokens} \
        --save_result_path ${save_result_path} > logs/evaluate_spqr_quant_qwen_chat.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1
fi
