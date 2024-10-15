#!/bin/bash

model_size=14B

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    export LD_LIBRARY_PATH=/home/work/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
else
    export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
fi

conda activate quant

cd ../../

mkdir -p code/scripts/logs

CUDA_VISIBLE_DEVICES=2 nohup python -u code/llm_int8_quant_qwen_chat_inference.py \
    --model_name llm_int8_01_Qwen-${model_size}-Chat \
    --constraint_types example mixed \
    --checkpoint_path ${checkpoint_path} > code/scripts/logs/llm_int8_quant_qwen_chat_inference.llm_int8_01_Qwen-${model_size}-Chat.logs 2>&1 &
