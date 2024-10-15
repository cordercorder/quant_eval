#!/bin/bash
# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    export TRANSFORMERS_CACHE=${work_dir}/.cache/huggingface
    export HF_MODULES_CACHE=${work_dir}/.cache/huggingface
# 02
else
    work_dir=/home/jinrenren
fi

source ${work_dir}/miniconda3/etc/profile.d/conda.sh

conda activate quant

export TRANSFORMERS_CACHE=${work_dir}/.cache/transformers
export HF_MODULES_CACHE=${work_dir}/.cache/transformers/huggingface
export TRITON_CACHE_DIR=${work_dir}/.cache
export TMPDIR=${work_dir}/tmp

split=${1}
model_size=${2}
num_bits=${3}

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
# spqr_w3g16_Qwen-14B-Chat
quant_checkpoint_path=${work_dir}/pretrained_models.quant/spqr_${num_bits}g16_Qwen-${model_size}-Chat
eval_data_path=${work_dir}/datasets/ceval
save_result_dir=${work_dir}/quant_eval_results/ceval.${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat

nohup python -u ../evaluate_spqr_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --quant_checkpoint_path ${quant_checkpoint_path} \
    --eval_data_path ${eval_data_path} \
    --save_result_dir ${save_result_dir} \
    --split ${split} \
    --pid -1 > logs/evaluate_spqr_quant_qwen_chat.${split}.spqr_${num_bits}g16_Qwen-${model_size}-Chat.logs 2>&1 &

echo last pid $!