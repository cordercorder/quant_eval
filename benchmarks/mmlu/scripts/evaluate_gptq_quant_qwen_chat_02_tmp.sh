#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/jinrenren/pretrained_models/Qwen-7B-Chat \
    --quant_checkpoint_path /home/jinrenren/pretrained_models.quant/gptq_02_Qwen-7B-Chat \
    --eval_data_path /home/jinrenren/datasets/mmlu \
    --save_result_dir /home/jinrenren/quant_eval_results/mmlu.gptq_02_Qwen-7B-Chat
