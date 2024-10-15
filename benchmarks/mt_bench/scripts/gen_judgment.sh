#!/bin/bash

# nohup python ../gen_judgment.py \
#     --judge-file /mnt/d/codes/FastChat/fastchat/llm_judge/data/judge_prompts.jsonl \
#     --question-file /mnt/d/codes/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
#     --answer-dir /mnt/d/codes/quant_eval/benchmarks/mt_bench/scripts/model_answer \
#     --ref-answer-dir /mnt/d/codes/FastChat/fastchat/llm_judge/data/mt_bench/reference_answer > gen_judgment_01.logs 2>&1 &

nohup python ../gen_judgment.py \
    --judge-file /mnt/d/codes/FastChat/fastchat/llm_judge/data/judge_prompts.jsonl \
    --model-list \
        gptq_03_Qwen-14B-Chat \
        gptq_02_Qwen-7B-Chat \
        gptq_02_Qwen-72B-Chat \
        gptq_02_Qwen-14B-Chat \
        gptq_01_Qwen-7B-Chat \
        gptq_01_Qwen-14B-Chat \
        gptq_04_Qwen-14B-Chat \
        gptq_03_Qwen-7B-Chat \
        gptq_03_Qwen-72B-Chat \
        test.spqr_w3g16_Qwen-72B-Chat \
        test.spqr_w2g16_Qwen-72B-Chat \
        gptq_04_Qwen-7B-Chat \
    --question-file /mnt/d/codes/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
    --answer-dir /mnt/d/codes/quant_eval/benchmarks/mt_bench/scripts/model_answer \
    --ref-answer-dir /mnt/d/codes/FastChat/fastchat/llm_judge/data/mt_bench/reference_answer > gen_judgment_02.logs 2>&1 &
