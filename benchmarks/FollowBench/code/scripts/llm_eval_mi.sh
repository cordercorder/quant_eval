#!/bin/bash

model_size=${1}
num_bits=${2}
wait_pid=${3:--1}

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
    data_path=/home/work/jinrenren/quant_eval/benchmarks/FollowBench/data
    api_output_path=/home/work/jinrenren/quant_eval/benchmarks/FollowBench/api_output
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
    data_path=/home/jinrenren/quant_eval/benchmarks/FollowBench/data
    api_output_path=/home/jinrenren/quant_eval/benchmarks/FollowBench/api_output
fi

conda activate quant

mkdir -p logs


model_path=spqr_${num_bits}g16_Qwen-${model_size}-Chat
logs_file_name=llm_eval_mi.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs

echo "model_path: ${model_path}"


nohup python -u ../llm_eval_mi.py \
    --model_path ${model_path} \
    --data_path ${data_path} \
    --api_output_path ${api_output_path} \
    --pid ${wait_pid} > logs/${logs_file_name} 2>&1 &
echo $!