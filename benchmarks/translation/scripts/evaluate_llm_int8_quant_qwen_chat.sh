#!/bin/bash

export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=2,3 nohup python -u ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest \
    --save_result_path /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.llm_int8_01_Qwen-72B-Chat.devtest \
    --source_lang English \
    --target_lang Chinese > evaluate_llm_int8_quant_qwen_chat.llm_int8_01_Qwen-72B-Chat.logs 2>&1 &
