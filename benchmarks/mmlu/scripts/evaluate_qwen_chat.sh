#!/bin/bash

CUDA_VISIBLE_DEVICES=3,4 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/jinrenren/datasets/mmlu \
    --save_result_dir /home/jinrenren/quant_eval_results/mmlu.bf16.Qwen-72B-Chat \
    --torch_dtype bf16 \
    --device_map auto > evaluate_qwen_chat.bf16.Qwen-72B-Chat.logs 2>&1 &

CUDA_VISIBLE_DEVICES=5,6 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/jinrenren/datasets/mmlu \
    --save_result_dir /home/jinrenren/quant_eval_results/mmlu.fp16.Qwen-72B-Chat \
    --torch_dtype fp16 \
    --device_map auto > evaluate_qwen_chat.fp16.Qwen-72B-Chat.logs 2>&1 &
