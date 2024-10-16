#!/bin/bash

mkdir -p ../results

python -u ../merge_results.py \
    --save_result_dirs \
        /home/jinrenren/quant_eval_results/bbq/bf16.Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/bf16.Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/bf16.Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/fp16.Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/fp16.Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/fp16.Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_01_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_01_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_01_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_02_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_02_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_02_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_03_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_03_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_03_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_04_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_04_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/gptq_04_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/llm_int8_01_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/llm_int8_01_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/llm_int8_01_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w2g16_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w2g16_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w2g16_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w3g16_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w3g16_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w3g16_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w4g16_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w4g16_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w4g16_Qwen-7B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w8g16_Qwen-14B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w8g16_Qwen-72B-Chat \
        /home/jinrenren/quant_eval_results/bbq/spqr_w8g16_Qwen-7B-Chat \
    --model_names \
        bf16.Qwen-14B-Chat \
        bf16.Qwen-72B-Chat \
        bf16.Qwen-7B-Chat \
        fp16.Qwen-14B-Chat \
        fp16.Qwen-72B-Chat \
        fp16.Qwen-7B-Chat \
        gptq_01_Qwen-14B-Chat \
        gptq_01_Qwen-72B-Chat \
        gptq_01_Qwen-7B-Chat \
        gptq_02_Qwen-14B-Chat \
        gptq_02_Qwen-72B-Chat \
        gptq_02_Qwen-7B-Chat \
        gptq_03_Qwen-14B-Chat \
        gptq_03_Qwen-72B-Chat \
        gptq_03_Qwen-7B-Chat \
        gptq_04_Qwen-14B-Chat \
        gptq_04_Qwen-72B-Chat \
        gptq_04_Qwen-7B-Chat \
        llm_int8_01_Qwen-14B-Chat \
        llm_int8_01_Qwen-72B-Chat \
        llm_int8_01_Qwen-7B-Chat \
        spqr_w2g16_Qwen-14B-Chat \
        spqr_w2g16_Qwen-72B-Chat \
        spqr_w2g16_Qwen-7B-Chat \
        spqr_w3g16_Qwen-14B-Chat \
        spqr_w3g16_Qwen-72B-Chat \
        spqr_w3g16_Qwen-7B-Chat \
        spqr_w4g16_Qwen-14B-Chat \
        spqr_w4g16_Qwen-72B-Chat \
        spqr_w4g16_Qwen-7B-Chat \
        spqr_w8g16_Qwen-14B-Chat \
        spqr_w8g16_Qwen-72B-Chat \
        spqr_w8g16_Qwen-7B-Chat \
    --merged_result_dir \
        ../results
