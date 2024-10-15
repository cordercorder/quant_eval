#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

model_size=${1}
num_bits=${2}

start_idx=${4}

if [[ ${#start_idx} -eq 0 ]]; then
    start_idx=0
fi

declare -A bits_map
bits_map[8]="01"
bits_map[4]="02"
bits_map[3]="03"
bits_map[2]="04"

dataset_name=pg19
filed_name=text

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    checkpoint_path=/home/work/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/work/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/work/jinrenren/datasets/pg19/test
    save_result_path=/home/work/jinrenren/quant_eval_results/memory/${dataset_name}/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
else
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/pg19/test
    save_result_path=/home/jinrenren/quant_eval_results/memory/${dataset_name}/gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat
fi

mkdir -p logs

# 7 * 8 * 16 * 2 = 1792

counter=0

for batch_size in 1; do
    for((input_length=8;input_length<=2048;input_length+=256)); do
        for((num_new_tokens=8;num_new_tokens<=2048;num_new_tokens+=128)); do

            if [[ ${counter} -lt ${start_idx} ]]; then
                continue
            fi

            echo "${counter} start!" >> logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs

            echo "batch_size: ${batch_size}, input_length: ${input_length}, num_new_tokens: ${num_new_tokens}" >> logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs

            echo "no_flash_attn start!" >> logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs

            CUDA_VISIBLE_DEVICES=${3} python -u ../evaluate_gptq_quant_qwen_chat.py \
                --checkpoint_path ${checkpoint_path} \
                --quant_checkpoint_path ${quant_checkpoint_path} \
                --eval_data_path ${eval_data_path} \
                --filed_name ${filed_name} \
                --batch_size ${batch_size} \
                --input_length ${input_length} \
                --num_new_tokens ${num_new_tokens} \
                --save_result_path ${save_result_path} >> logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1
            
            echo "${counter} end!" >> logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs
            
            counter=$((counter+1))
            # echo "with_flash_attn start!"

            # CUDA_VISIBLE_DEVICES=${3} python -u ../evaluate_gptq_quant_qwen_chat.py \
            #     --checkpoint_path ${checkpoint_path} \
            #     --quant_checkpoint_path ${quant_checkpoint_path} \
            #     --eval_data_path ${eval_data_path} \
            #     --filed_name ${filed_name} \
            #     --batch_size ${batch_size} \
            #     --input_length ${input_length} \
            #     --num_new_tokens ${num_new_tokens} \
            #     --use_flash_attn \
            #     --save_result_path ${save_result_path} >> logs/evaluate_gptq_quant_qwen_chat.gptq_${bits_map[${num_bits}]}_Qwen-${model_size}-Chat.logs 2>&1

        done
    done
done
