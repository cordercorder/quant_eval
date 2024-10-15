#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
torch_dtype=${2}

dataset_name=cnn_dailymail
document_key=article
summary_key=highlights

split=test

mkdir -p logs

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/${dataset_name}
    save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.${torch_dtype}.Qwen-${model_size}-Chat
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/${dataset_name}
    save_result_path=/home/jinrenren/quant_eval_results/${dataset_name}/${split}.${torch_dtype}.Qwen-${model_size}-Chat
fi

CUDA_VISIBLE_DEVICES=${3} nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} \
    --document_key ${document_key} \
    --summary_key ${summary_key} \
    --torch_dtype ${torch_dtype} > logs/evaluate_qwen_chat.${dataset_name}.${split}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
