#!/bin/bash

source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/transformers
export HF_MODULES_CACHE=/home/work/jinrenren/.cache/transformers/huggingface

split=val

# CUDA_VISIBLE_DEVICES=1,7 nohup python -u ../evaluate_qwen_chat.py \
#     --checkpoint_path /home/jinrenren/pretrained_models/Qwen-72B-Chat \
#     --eval_data_path /home/jinrenren/datasets/ceval \
#     --save_result_dir /home/jinrenren/quant_eval_results/ceval.${split}.bf16.Qwen-72B-Chat \
#     --split ${split} \
#     --torch_dtype bf16 > evaluate_qwen_chat.${split}.bf16.Qwen-72B-Chat.logs 2>&1 &

CUDA_VISIBLE_DEVICES=6,7 nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/ceval \
    --save_result_dir /home/work/jinrenren/quant_eval_results/ceval.${split}.fp16.Qwen-72B-Chat \
    --split ${split} \
    --torch_dtype fp16 > evaluate_qwen_chat.${split}.fp16.Qwen-72B-Chat.logs 2>&1 &
