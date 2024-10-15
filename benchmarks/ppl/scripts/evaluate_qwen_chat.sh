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

dataset_name=alpaca_gpt4_data.json


model_size=${1}
torch_dtype=${2}
wikitext_batch_size=${3:-16}
c4_batch_size=${4:-1}
ptb_batch_size=${5:-1}
wait_pid=${6:--1}

checkpoint_path=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
save_result_path=${work_dir}/quant_eval_results/ppl/1_Qwen-${model_size}-Chat.${torch_dtype}.json
mkdir -p ${work_dir}/quant_eval_results/ppl/


c4_data_path=${work_dir}/datasets/ppl/c4/en.c4-validation.00000-of-00008
wikitext_data_path=${work_dir}/datasets/ppl/wikitext/wikitext2_test
ptb_data_path=${work_dir}/datasets/ppl/ptb/ptb_test.bin


mkdir -p logs

nohup python ../evaluate_qwen_chat.py \
    --checkpoint_path ${checkpoint_path} \
    --wikitext --c4 --ptb \
    --wikitext_batch_size ${wikitext_batch_size} --c4_batch_size ${c4_batch_size} --ptb_batch_size ${ptb_batch_size} \
    --c4_data_path ${c4_data_path} \
    --wikitext_data_path ${wikitext_data_path} \
    --ptb_data_path ${ptb_data_path} \
    --torch_dtype ${torch_dtype} \
    --save_result_path ${save_result_path} \
    --pid ${wait_pid} > logs/1_evaluate_qwen_chat.${torch_dtype}.Qwen-${model_size}-Chat.logs 2>&1 &
echo $!
