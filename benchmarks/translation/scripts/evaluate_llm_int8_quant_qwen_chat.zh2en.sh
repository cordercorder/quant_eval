#!/bin/bash

export LD_LIBRARY_PATH=/home/work/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH

rm -rf /home/work/jinrenren/.cache/*

export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/huggingface/transformers
export HF_MODULES_CACHE=/home/work/jinrenren/.cache/huggingface

CUDA_VISIBLE_DEVICES=0 nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-14B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.llm_int8_01_Qwen-14B-Chat.devtest \
    --source_lang Chinese \
    --target_lang English > evaluate_llm_int8_quant_qwen_chat.zh2en.llm_int8_01_Qwen-14B-Chat.logs 2>&1 &
