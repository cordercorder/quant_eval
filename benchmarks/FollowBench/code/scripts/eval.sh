#!/bin/bash

# model_size=${1}
# num_bits=${2}

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    data_path=/home/work/jinrenren/quant_eval/benchmarks/FollowBench/data
    api_output_path=/home/work/jinrenren/quant_eval/benchmarks/FollowBench/api_output
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    data_path=/home/jinrenren/quant_eval/benchmarks/FollowBench/data
    api_output_path=/home/jinrenren/quant_eval/benchmarks/FollowBench/api_output
fi

declare -A model_path_group

mkdir -p logs
# spqr_w3g16_Qwen-7B-Chat
# spqr_w3g16_Qwen-7B-Chat \
# spqr_w4g16_Qwen-7B-Chat \
# spqr_w8g16_Qwen-7B-Chat \
# spqr_w3g16_Qwen-14B-Chat \
# spqr_w4g16_Qwen-14B-Chat \
# spqr_w8g16_Qwen-14B-Chat \
# spqr_w3g16_Qwen-72B-Chat \
# spqr_w4g16_Qwen-72B-Chat \
# spqr_w8g16_Qwen-72B-Chat \

# spqr_w2g16_Qwen-14B
# spqr_w2g16_Qwen-72B
# spqr_w2g16_Qwen-7B
# spqr_w3g16_Qwen-14B
# spqr_w3g16_Qwen-72B
# spqr_w3g16_Qwen-7B
# spqr_w4g16_Qwen-14B
# spqr_w4g16_Qwen-72B
# spqr_w4g16_Qwen-7B
# spqr_w8g16_Qwen-14B
# spqr_w8g16_Qwen-72B
# spqr_w8g16_Qwen-7B


nohup python -u ../eval.py \
    --model_paths \
        spqr_w2g16_Qwen-7B-Chat \
        spqr_w3g16_Qwen-7B-Chat \
        spqr_w4g16_Qwen-7B-Chat \
        spqr_w8g16_Qwen-7B-Chat \
        spqr_w2g16_Qwen-14B-Chat \
        spqr_w3g16_Qwen-14B-Chat \
        spqr_w4g16_Qwen-14B-Chat \
        spqr_w8g16_Qwen-14B-Chat \
        spqr_w2g16_Qwen-72B-Chat \
        spqr_w3g16_Qwen-72B-Chat \
        spqr_w4g16_Qwen-72B-Chat \
        spqr_w8g16_Qwen-72B-Chat \
    --data_path ${data_path} \
    --api_output_path ${api_output_path} > logs/eval.spqr_w2,3,4,8_g16_Qwen-Chat.logs 2>&1 &
