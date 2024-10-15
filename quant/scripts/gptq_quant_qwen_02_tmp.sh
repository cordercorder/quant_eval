#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

quantized_model_dir=/home/work/jinrenren/pretrained_models.quant/gptq_02_Qwen-14B-Chat

mkdir -p ${quantized_model_dir}

nohup python -u ../gptq_quant_qwen.py \
    --pretrained_model_name_or_path /home/work/jinrenren/pretrained_models/Qwen-14B-Chat \
    --fast_tokenizer \
    --quantized_model_dir ${quantized_model_dir} \
    --bits 4 \
    --use_triton \
    --per_gpu_max_memory 80 \
    --data_path /home/work/jinrenren/datasets/alpaca_gpt4_data.json > ${quantized_model_dir}/gptq_quant_qwen.logs 2>&1 &
