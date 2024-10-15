#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
dataset_name=alpaca_eval_eval

split=test

mkdir -p logs

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    export LD_LIBRARY_PATH=/home/work/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/tatsu-lab/alpaca_eval/alpaca_eval_eval
    save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}/${split}.llm_int8_01_Qwen-${model_size}-Chat
else
    export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/tatsu-lab/alpaca_eval/alpaca_eval_eval
    save_result_path=/home/jinrenren/quant_eval_results/${dataset_name}/${split}.llm_int8_01_Qwen-${model_size}-Chat
fi

CUDA_VISIBLE_DEVICES=${2} nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_path ${save_result_path} > logs/evaluate_llm_int8_quant_qwen_chat.${dataset_name}.${split}.llm_int8_01_Qwen-${model_size}-Chat.logs 2>&1 &
