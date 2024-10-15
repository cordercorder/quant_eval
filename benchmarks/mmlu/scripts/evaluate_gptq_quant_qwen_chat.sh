#!/bin/bash

source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/transformers
export HF_MODULES_CACHE=/home/work/jinrenren/.cache/transformers/huggingface

CUDA_VISIBLE_DEVICES=5 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_04_Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/mmlu \
    --max_new_tokens 20 \
    --save_result_dir /home/work/jinrenren/quant_eval_results/mmlu.gptq_04_Qwen-72B-Chat >> evaluate_gptq_quant_qwen_chat.gptq_04_Qwen-72B-Chat.logs 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
#     --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
#     --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_02_Qwen-72B-Chat \
#     --eval_data_path /home/work/jinrenren/datasets/mmlu \
#     --save_result_dir /home/work/jinrenren/quant_eval_results/mmlu.gptq_02_Qwen-72B-Chat > evaluate_gptq_quant_qwen_chat.gptq_02_Qwen-72B-Chat.logs 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
#     --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
#     --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_03_Qwen-72B-Chat \
#     --eval_data_path /home/work/jinrenren/datasets/mmlu \
#     --save_result_dir /home/work/jinrenren/quant_eval_results/mmlu.gptq_03_Qwen-72B-Chat > evaluate_gptq_quant_qwen_chat.gptq_03_Qwen-72B-Chat.logs 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
#     --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
#     --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_04_Qwen-72B-Chat \
#     --eval_data_path /home/work/jinrenren/datasets/mmlu \
#     --save_result_dir /home/work/jinrenren/quant_eval_results/mmlu.gptq_04_Qwen-72B-Chat > evaluate_gptq_quant_qwen_chat.gptq_04_Qwen-72B-Chat.logs 2>&1 &
