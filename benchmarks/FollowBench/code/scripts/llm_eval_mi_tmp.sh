#!/bin/bash

model_size=14B
torch_dtype=none
num_bits=8
llm_int8=0

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

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

if [[ ${num_bits} = "0" ]] && [[ ${torch_dtype} != "none" ]]; then
    model_path=${torch_dtype}.Qwen-${model_size}-Chat
    logs_file_name=llm_eval_mi.${torch_dtype}.Qwen-${model_size}-Chat.logs
elif [[ ${num_bits} = "8" ]] && [[ ${llm_int8} = "1" ]]; then
    model_path=llm_int8_01_Qwen-${model_size}-Chat
    logs_file_name=llm_eval_mi.llm_int8_01_Qwen-${model_size}-Chat.logs
else
    model_path=gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    logs_file_name=llm_eval_mi.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs
fi

echo "model_path: ${model_path}"


nohup python -u ../llm_eval_mi.py \
    --model_path ${model_path} \
    --constraint_types content \
    --data_path ${data_path} \
    --api_output_path ${api_output_path} > logs/${logs_file_name} 2>&1 &
