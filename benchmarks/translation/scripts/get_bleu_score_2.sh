#!/bin/bash

outputs_dir=.tmp
reference_path=/home/jinrenren/datasets/flores200_dataset/devtest/zho_Hans.devtest

# reference_path=/home/jinrenren/datasets/flores200_dataset/devtest/eng_Latn.devtest

mkdir -p ${outputs_dir}

# inputs="
# /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.llm_int8_01_Qwen-72B-Chat.devtest \
# "

inputs="
/home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.spqr_w2g16_Qwen-72B-Chat.devtest \
"

# inputs="
# /home/jinrenren/quant_eval_results/flores200_dataset/zho_Hans.llm_int8_01_Qwen-7B-Chat.devtest
# "

outputs="
${outputs_dir}/zho_Hans.spqr_w2g16_Qwen-72B-Chat.spm_encode.devtest \
"

# outputs="
# ${outputs_dir}/zho_Hans.llm_int8_01_Qwen-7B-Chat.devtest
# "

python ../spm_encode.py \
    --model /home/jinrenren/datasets/flores200_dataset/flores200_sacrebleu_tokenizer_spm.model \
    --output_format piece \
    --inputs \
        ${inputs} \
        ${reference_path} \
    --outputs \
        ${outputs} \
        ${outputs_dir}/zho_Hans.reference.devtest


for output in ${outputs}; do
    sacrebleu ${outputs_dir}/zho_Hans.reference.devtest -w 2 -i ${output}
done
