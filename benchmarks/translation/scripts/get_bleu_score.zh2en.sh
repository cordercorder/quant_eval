#!/bin/bash

outputs_dir=.tmp
# reference_path=/home/work/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest

reference_path=/home/work/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest

mkdir -p ${outputs_dir}

# inputs="
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.llm_int8_01_Qwen-14B-Chat.devtest \
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.Qwen-72B-Chat.bf16.devtest \
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.Qwen-72B-Chat.fp16.devtest \
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_01_Qwen-72B-Chat.devtest \
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_02_Qwen-72B-Chat.devtest \
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_03_Qwen-72B-Chat.devtest \
# /home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.gptq_04_Qwen-72B-Chat.devtest \
# "

inputs="
/home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.spqr_w3g16_Qwen-72B-Chat.devtest \
/home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.spqr_w4g16_Qwen-72B-Chat.devtest \
/home/work/jinrenren/quant_eval_results/flores200_dataset/zho_Hans-eng_Latn.spqr_w8g16_Qwen-72B-Chat.devtest \
"


# inputs="
# /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.llm_int8_01_Qwen-7B-Chat.devtest
# "

outputs="
${outputs_dir}/zho_Hans-eng_Latn.spqr_w3g16_Qwen-72B-Chat.spm_encode.devtest \
${outputs_dir}/zho_Hans-eng_Latn.spqr_w4g16_Qwen-72B-Chat.spm_encode.devtest \
${outputs_dir}/zho_Hans-eng_Latn.spqr_w8g16_Qwen-72B-Chat.spm_encode.devtest \
"

python ../spm_encode.py \
    --model /home/work/jinrenren/datasets/flores200_dataset/flores200_sacrebleu_tokenizer_spm.model \
    --output_format piece \
    --inputs \
        ${inputs} \
        ${reference_path} \
    --outputs \
        ${outputs} \
        ${outputs_dir}/eng_Latn.reference.devtest


for output in ${outputs}; do
    sacrebleu ${outputs_dir}/eng_Latn.reference.devtest -w 2 -i ${output}
done
