#!/bin/bash

export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/huggingface
export HF_MODULES_CACHE=/home/work/jinrenren/.cache/huggingface

CUDA_VISIBLE_DEVICES=7 nohup /home/work/jinrenren/miniconda3/envs/quant/bin/python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-7B-Chat \
    --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_04_Qwen-7B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.gptq_04_Qwen-7B-Chat.devtest \
    --source_lang Chinese \
    --target_lang English > evaluate_gptq_quant_qwen_chat.zh2en.gptq_04_Qwen-7B-Chat.logs 2>&1 &
