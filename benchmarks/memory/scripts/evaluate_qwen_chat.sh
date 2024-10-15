#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
device=${2}

torch_dtype=bf16

dataset_name=pg19
filed_name=text

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/pg19/test
    save_result_path=/home/work/jinrenren/quant_eval_results/memory/${dataset_name}/${torch_dtype}.Qwen-${model_size}-Chat
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/pg19/test
    save_result_path=/home/jinrenren/quant_eval_results/memory/${dataset_name}/${torch_dtype}.Qwen-${model_size}-Chat
fi

mkdir -p logs

# 7 * 8 * 16 * 2 = 1792

batch_size=1
input_length=256
num_new_tokens=512

if [[ ${#device} -eq 0 ]]; then
    python -u ../evaluate_qwen_chat.py \
        --checkpoint_path ${checkpoint_path} \
        --eval_data_path ${eval_data_path} \
        --filed_name ${filed_name} \
        --torch_dtype ${torch_dtype} \
        --batch_size ${batch_size} \
        --input_length ${input_length} \
        --num_new_tokens ${num_new_tokens} \
        --save_result_path ${save_result_path} > logs/evaluate_qwen_chat.${dataset_name}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1
else
    CUDA_VISIBLE_DEVICES=${device} python -u ../evaluate_qwen_chat.py \
        --checkpoint_path ${checkpoint_path} \
        --eval_data_path ${eval_data_path} \
        --filed_name ${filed_name} \
        --torch_dtype ${torch_dtype} \
        --batch_size ${batch_size} \
        --input_length ${input_length} \
        --num_new_tokens ${num_new_tokens} \
        --save_result_path ${save_result_path} > logs/evaluate_qwen_chat.${dataset_name}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1
fi
