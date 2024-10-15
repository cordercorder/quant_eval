#!/bin/bash

source /home/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

export TRANSFORMERS_CACHE=/home/jinrenren/.cache/transformers
export HF_MODULES_CACHE=/home/jinrenren/.cache/transformers/huggingface

split=test

CUDA_VISIBLE_DEVICES=6 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-7B-Chat \
    --eval_data_path /home/jinrenren/datasets/ceval \
    --save_result_dir /home/jinrenren/quant_eval_results/ceval.${split}.bf16.Qwen-7B-Chat \
    --split ${split} \
    --torch_dtype bf16 > evaluate_qwen_chat.${split}.bf16.Qwen-7B-Chat.logs 2>&1 &


CUDA_VISIBLE_DEVICES=6 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-7B-Chat \
    --eval_data_path /home/jinrenren/datasets/ceval \
    --save_result_dir /home/jinrenren/quant_eval_results/ceval.${split}.fp16.Qwen-7B-Chat \
    --split ${split} \
    --torch_dtype fp16 > evaluate_qwen_chat.${split}.fp16.Qwen-7B-Chat.logs 2>&1 &



split=val

CUDA_VISIBLE_DEVICES=0 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-14B-Chat \
    --eval_data_path /home/jinrenren/datasets/ceval \
    --save_result_dir /home/jinrenren/quant_eval_results/ceval.${split}.bf16.Qwen-14B-Chat \
    --split ${split} \
    --torch_dtype bf16 > evaluate_qwen_chat.${split}.bf16.Qwen-14B-Chat.logs 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-14B-Chat \
    --eval_data_path /home/jinrenren/datasets/ceval \
    --save_result_dir /home/jinrenren/quant_eval_results/ceval.${split}.fp16.Qwen-14B-Chat \
    --split ${split} \
    --torch_dtype fp16 > evaluate_qwen_chat.${split}.fp16.Qwen-14B-Chat.logs 2>&1 &
