#!/bin/bash

export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=7 nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-7B-Chat \
    --eval_data_path /home/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest \
    --save_result_path /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.llm_int8_01_Qwen-7B-Chat.devtest \
    --source_lang Chinese \
    --target_lang English > evaluate_llm_int8_quant_qwen_chat.zh2en.llm_int8_01_Qwen-7B-Chat.logs 2>&1 &
