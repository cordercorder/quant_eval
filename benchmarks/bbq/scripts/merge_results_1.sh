#!/bin/bash

mkdir -p ../results_1

python -u ../merge_results_1.py \
    --save_result_dirs \
        /home/jinrenren/jinrenren-starfs-data/quant_eval_results/bbq/spqr_w2g16_Qwen-72B-Chat \
    --model_names \
        spqr_w2g16_Qwen-72B-Chat \
    --merged_result_dir \
        ../results_1
