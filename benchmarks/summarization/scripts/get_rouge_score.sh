#!/bin/bash

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    source /home/work/jinrenren/miniconda3/etc/profile.d/conda.sh
else
    source /home/jinrenren/miniconda3/etc/profile.d/conda.sh
fi

conda activate quant

dataset_name=${1}

if [[ `hostname` = "tj5-g8-a100-v6-tj5-translator03.kscn" ]]; then
    save_result_path=/home/work/jinrenren/quant_eval_results/${dataset_name}
else
    save_result_path=/home/jinrenren/quant_eval_results/${dataset_name}
fi

args=""
for file_name in `ls ${save_result_path}`; do
    file_path=${save_result_path}/${file_name}
    args+="${file_path} "
done


python -u ../get_rouge_score.py \
    -i ${args}
