#!/bin/bash

# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    export TRANSFORMERS_CACHE=${work_dir}/.cache/huggingface
    export HF_MODULES_CACHE=${work_dir}/.cache/huggingface
    export LD_LIBRARY_PATH=/home/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
else
    # 02
    work_dir=/home/jinrenren
    export LD_LIBRARY_PATH=/home/work/jinrenren/miniconda3/envs/quant/lib:$LD_LIBRARY_PATH
fi

source ${work_dir}/miniconda3/etc/profile.d/conda.sh
conda activate quant

model_size=${1}
wait_pid=${2:--1}

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
save_result_path=${work_dir}/quant_eval_results/ppl/llm_int8_01_Qwen-${model_size}-Chat.json
mkdir -p ${work_dir}/quant_eval_results/ppl/

c4_data_path=${work_dir}/datasets/ppl/c4/en.c4-validation.00000-of-00008
wikitext_data_path=${work_dir}/datasets/ppl/wikitext/wikitext2_test
ptb_data_path=${work_dir}/datasets/ppl/ptb/ptb_test.bin

mkdir -p logs

nohup python ../evaluate_llm_int8_quant_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --data_paths \
        ${work_dir}/datasets/ppl/wikitext/wikitext2_test \
        ${work_dir}/datasets/ppl/ptb/ptb_test.bin \
        ${work_dir}/datasets/ppl/c4/en.c4-validation.00000-of-00008 \
    --fileds \
        text \
        sentence \
        text \
    --num_samples \
        -1 \
        -1 \
        1100 \
    --save_result_path ${save_result_path} \
    --pid ${wait_pid} > logs/evaluate_llm_int8_quant_qwen_chat.llm_int8_01_Qwen-${model_size}-Chat.logs 2>&1 &
echo $!
