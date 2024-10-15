#!/bin/bash

CUDA_VISIBLE_DEVICES=3,5 python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest \
    --save_result_path /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.Qwen-72B-Chat.fp16.devtest \
    --source_lang Chinese \
    --target_lang English \
    --torch_dtype fp16 > evaluate_qwen_chat.Qwen-72B-Chat.zh2en.fp16.logs 2>&1

CUDA_VISIBLE_DEVICES=3,5 python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest \
    --save_result_path /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.Qwen-72B-Chat.bf16.devtest \
    --source_lang Chinese \
    --target_lang English \
    --torch_dtype bf16 > evaluate_qwen_chat.Qwen-72B-Chat.zh2en.bf16.logs 2>&1
