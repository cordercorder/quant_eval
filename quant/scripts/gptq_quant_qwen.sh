#!/bin/bash

quantized_model_dir=/home/work/jinrenren/pretrained_models.quant/gptq_02_Qwen-72B-Chat

mkdir -p ${quantized_model_dir}

CUDA_VISIBLE_DEVICES=1,5,6 python -u ../gptq_quant_qwen.py \
    --pretrained_model_name_or_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --fast_tokenizer \
    --quantized_model_dir ${quantized_model_dir} \
    --bits 4 \
    --per_gpu_max_memory 80 \
    --data_path /home/work/jinrenren/datasets/alpaca_gpt4_data.json > ${quantized_model_dir}/gptq_quant_qwen.logs 2>&1



quantized_model_dir=/home/work/jinrenren/pretrained_models.quant/gptq_03_Qwen-72B-Chat

mkdir -p ${quantized_model_dir}

CUDA_VISIBLE_DEVICES=1,5,6 python -u ../gptq_quant_qwen.py \
    --pretrained_model_name_or_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --fast_tokenizer \
    --quantized_model_dir ${quantized_model_dir} \
    --bits 3 \
    --per_gpu_max_memory 80 \
    --data_path /home/work/jinrenren/datasets/alpaca_gpt4_data.json > ${quantized_model_dir}/gptq_quant_qwen.logs 2>&1



quantized_model_dir=/home/work/jinrenren/pretrained_models.quant/gptq_04_Qwen-72B-Chat

mkdir -p ${quantized_model_dir}

CUDA_VISIBLE_DEVICES=1,5,6 python -u ../gptq_quant_qwen.py \
    --pretrained_model_name_or_path /home/work/jinrenren/pretrained_models/Qwen-72B-Chat \
    --fast_tokenizer \
    --quantized_model_dir ${quantized_model_dir} \
    --bits 2 \
    --per_gpu_max_memory 80 \
    --data_path /home/work/jinrenren/datasets/alpaca_gpt4_data.json > ${quantized_model_dir}/gptq_quant_qwen.logs 2>&1
