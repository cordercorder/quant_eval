#!/bin/bash

source /home/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH

export TRANSFORMERS_CACHE=/home/jinrenren/.cache/huggingface/transformers
export HF_MODULES_CACHE=/home/jinrenren/.cache/huggingface

split=${1}
model_size=${2}

checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
eval_data_path=/home/jinrenren/datasets/ceval
save_result_dir=/home/jinrenren/quant_eval_results/ceval.${split}.llm_int8_01_Qwen-${model_size}-Chat

CUDA_VISIBLE_DEVICES=${3} nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_dir ${save_result_dir} \
    --split ${split} > evaluate_llm_int8_quant_qwen_chat.${split}.llm_int8_01_Qwen-${model_size}-Chat.logs 2>&1 &
