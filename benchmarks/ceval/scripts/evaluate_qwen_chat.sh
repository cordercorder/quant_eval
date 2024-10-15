#!/bin/bash

source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh

conda activate quant

export TRANSFORMERS_CACHE=/home/work/jinrenren/.cache/transformers
export HF_MODULES_CACHE=/home/work/jinrenren/.cache/transformers/huggingface
export TRITON_CACHE_DIR=/home/work/jinrenren/.cache
export TMPDIR=/home/work/jinrenren/tmp

split=${1}
model_size=${2}
torch_dtype=${3}

checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
eval_data_path=/home/work/jinrenren/datasets/ceval
save_result_dir=/home/work/jinrenren/quant_eval_results/ceval.${split}.${torch_dtype}.Qwen-${model_size}-Chat

CUDA_VISIBLE_DEVICES=${4} nohup python -u ../evaluate_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_dir ${save_result_dir} \
    --split ${split} \
    --torch_dtype ${torch_dtype} > evaluate_qwen_chat.${split}.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
