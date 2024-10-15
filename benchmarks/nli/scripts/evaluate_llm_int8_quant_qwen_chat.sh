#!/bin/bash

model_size=${1}

dataset_name=snli
split=test

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    export LD_LIBRARY_PATH=/home/work/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/${dataset_name}
    save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.llm_int8_01_Qwen-${model_size}-Chat
else
    export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/${dataset_name}
    save_result_path=/home/jinrenren/quant_eval_results/${dataset_name}/${split}.llm_int8_01_Qwen-${model_size}-Chat
fi

conda activate quant

mkdir -p logs

CUDA_VISIBLE_DEVICES=${2} nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} \
    --split ${split} > logs/evaluate_llm_int8_quant_qwen_chat.${dataset_name}.${split}.llm_int8_01_Qwen-${model_size}-Chat.logs 2>&1 &
