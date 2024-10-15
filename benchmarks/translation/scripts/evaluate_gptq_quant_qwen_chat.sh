#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_01_Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_01_Qwen-72B-Chat.devtest \
    --source_lang English \
    --target_lang Chinese > evaluate_gptq_quant_qwen_chat.gptq_01_Qwen-72B-Chat.logs 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_02_Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_02_Qwen-72B-Chat.devtest \
    --source_lang English \
    --target_lang Chinese > evaluate_gptq_quant_qwen_chat.gptq_02_Qwen-72B-Chat.logs 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_03_Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_03_Qwen-72B-Chat.devtest \
    --source_lang English \
    --target_lang Chinese > evaluate_gptq_quant_qwen_chat.gptq_03_Qwen-72B-Chat.logs 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u ../evaluate_gptq_quant_qwen_chat.py \
    --checkpoint_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --quant_checkpoint_path /home/work/jinrenren/pretrained_models.quant/gptq_04_Qwen-72B-Chat \
    --eval_data_path /home/work/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest \
    --save_result_path /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_04_Qwen-72B-Chat.devtest \
    --source_lang English \
    --target_lang Chinese > evaluate_gptq_quant_qwen_chat.gptq_04_Qwen-72B-Chat.logs 2>&1 &
