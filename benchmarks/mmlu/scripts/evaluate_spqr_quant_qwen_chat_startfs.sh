#!/bin/bash

model_size=${1}
num_bits=${2}

# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
    eval_data_path=${work_dir}/datasets/mmlu
    source ${work_dir}/miniconda3/etc/profile.d/conda.sh
# 02
else
    work_dir=/home/jinrenren/jinrenren-starfs-data
    checkpoint_path=/home/jinrenren/pretrained_models/Qwen-${model_size}-Chat
    quant_checkpoint_path=/home/jinrenren/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
    eval_data_path=/home/jinrenren/datasets/mmlu
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

export TRANSFORMERS_CACHE=${work_dir}/.cache/huggingface
export HF_MODULES_CACHE=${work_dir}/.cache/huggingface

conda activate quant

save_result_dir=${work_dir}/quant_eval_results/mmlu.spqr_${num_bits}g16_Qwen-${model_size}-Chat

nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_dir ${save_result_dir} \
    --pid -1 > evaluate_spqr_quant_qwen_chat.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &
echo $!

# CUDA_VISIBLE_DEVICES=6  python -u ../evaluate_spqr_quant_qwen_chat.py \
#     --checkpoint_path ${work_dir}/pretrained_models/Qwen-7B-Chat \
#     --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_w3g16_Qwen-7B-Chat \
#     --eval_data_path ${work_dir}/datasets/mmlu \
#     --save_result_dir ${work_dir}/quant_eval_results/mmlu.spqr_w3g16_Qwen-7B-Chat > evaluate_spqr_quant_qwen_chat.spqr_w3g16_Qwen-7B-Chat.logs 2>&1


# CUDA_VISIBLE_DEVICES=6  python -u ../evaluate_spqr_quant_qwen_chat.py \
#     --checkpoint_path ${work_dir}/pretrained_models/Qwen-7B-Chat \
#     --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_w4g16_Qwen-7B-Chat \
#     --eval_data_path ${work_dir}/datasets/mmlu \
#     --save_result_dir ${work_dir}/quant_eval_results/mmlu.spqr_w4g16_Qwen-7B-Chat > evaluate_spqr_quant_qwen_chat.spqr_w4g16_Qwen-7B-Chat.logs 2>&1


# CUDA_VISIBLE_DEVICES=4 nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
#     --checkpoint_path ${work_dir}/pretrained_models/Qwen-7B-Chat \
#     --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_w8g16_Qwen-7B-Chat \
#     --eval_data_path ${work_dir}/datasets/mmlu \
#     --save_result_dir ${work_dir}/quant_eval_results/mmlu.spqr_w8g16_Qwen-7B-Chat \
#     --pid ${1:--1} > evaluate_spqr_quant_qwen_chat.spqr_w8g16_Qwen-7B-Chat.logs 2>&1 &


# CUDA_VISIBLE_DEVICES=6 nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
#     --checkpoint_path ${work_dir}/pretrained_models/Qwen-14B-Chat \
#     --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_w3g16_Qwen-14B-Chat \
#     --eval_data_path ${work_dir}/datasets/mmlu \
#     --save_result_dir ${work_dir}/quant_eval_results/mmlu.spqr_w3g16_Qwen-14B-Chat \
#     --pid ${!:--1} > evaluate_spqr_quant_qwen_chat.spqr_w3g16_Qwen-14B-Chat.logs 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
#     --checkpoint_path ${work_dir}/pretrained_models/Qwen-14B-Chat \
#     --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_w4g16_Qwen-14B-Chat \
#     --eval_data_path ${work_dir}/datasets/mmlu \
#     --save_result_dir ${work_dir}/quant_eval_results/mmlu.spqr_w4g16_Qwen-14B-Chat \
#     --pid ${!:--1} > evaluate_spqr_quant_qwen_chat.spqr_w4g16_Qwen-14B-Chat.logs 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
#     --checkpoint_path ${work_dir}/pretrained_models/Qwen-14B-Chat \
#     --quant_checkpoint_path ${work_dir}/pretrained_models.quant/spqr_w8g16_Qwen-14B-Chat \
#     --eval_data_path ${work_dir}/datasets/mmlu \
#     --save_result_dir ${work_dir}/quant_eval_results/mmlu.spqr_w8g16_Qwen-14B-Chat \
#     --pid ${2:--1}  > evaluate_spqr_quant_qwen_chat.spqr_w8g16_Qwen-14B-Chat.logs 2>&1 &

