#!/bin/bash

CUDA_VISIBLE_DEVICES=3,7 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.Qwen-72B-Chat.fp16.devtest \
    --source_lang English \
    --target_lang Chinese \
    --torch_dtype fp16 \
    --device_map auto > evaluate_qwen_chat.Qwen-72B-Chat.fp16.logs 2>&1 &
