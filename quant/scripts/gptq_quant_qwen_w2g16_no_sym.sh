#!/bin/bash

model_size=${1}
per_gpu_max_memory=${2}

quantized_model_dir=/home/work/jinrenren/pretrained_models.quant/gptq_w2g16_01_Qwen-${model_size}-Chat

mkdir -p ${quantized_model_dir}

CUDA_VISIBLE_DEVICES=${3} python -u ../gptq_quant_qwen.py \
    --pretrained_model_name_or_path /home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat \
    --fast_tokenizer \
    --quantized_model_dir ${quantized_model_dir} \
    --bits 2 \
    --group_size 16 \
    --no_sym \
    --no_true_sequential \
    --per_gpu_max_memory ${per_gpu_max_memory} \
    --data_path /home/work/jinrenren/datasets/alpaca_gpt4_data.json > ${quantized_model_dir}/gptq_quant_qwen.logs 2>&1
