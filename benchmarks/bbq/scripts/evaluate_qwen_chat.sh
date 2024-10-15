#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
torch_dtype=${2}
device=${3}

if [[ ${#device} -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES=${device}
fi

dataset_name=bbq

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_dir=/home/work/jinrenren/datasets/${dataset_name}
    save_result_dir=/home/work/jinrenren/quant_eval_results/${dataset_name}/${torch_dtype}.Qwen-${model_size}-Chat
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_dir=/home/jinrenren/datasets/${dataset_name}
    save_result_dir=/home/jinrenren/quant_eval_results/${dataset_name}/${torch_dtype}.Qwen-${model_size}-Chat
fi

mkdir -p logs

nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_dir ${eval_data_dir} \
    --save_result_dir ${save_result_dir} \
    --torch_dtype ${torch_dtype} > logs/evaluate_qwen_chat.${dataset_name}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
