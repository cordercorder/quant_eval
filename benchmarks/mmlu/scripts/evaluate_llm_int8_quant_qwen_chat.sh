#!/bin/bash

export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH

# export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/huggingface/transformers
# export HF_MODULES_CACHE=/home/work/jinrenren/.cache/huggingface

CUDA_VISIBLE_DEVICES=0 nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models//Qwen-7B-Chat \
    --eval_data_path /home/jinrenren/datasets/mmlu \
    --save_result_dir /home/jinrenren/quant_eval_results/mmlu.llm_int8_01_Qwen-7B-Chat > evaluate_llm_int8_quant_qwen_chat.llm_int8_01_Qwen-7B-Chat.logs 2>&1 &
