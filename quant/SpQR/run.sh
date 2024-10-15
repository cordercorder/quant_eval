# 03
if [ $(hostname) == tj5-g8-a100-v6-tj5-translator03.kscn ];then
    work_dir=/home/work/jinrenren
    export TRANSFORMERS_CACHE=${work_dir}/.cache/huggingface
    export HF_MODULES_CACHE=${work_dir}/.cache/huggingface
# 02
else
    work_dir=/home/jinrenren
fi

gpu=$1
wbits=$2
model_size=$3
MODEL_PATH=${work_dir}/pretrained_models/Qwen-${model_size}-Chat
CUSTOM_DATA_PATH=${work_dir}/datasets/alpaca_gpt4_data.json
quantized_model_dir=${work_dir}/pretrained_models.quant/spqr_w${2}g16_Qwen-${model_size}-Chat

mkdir -p ${quantized_model_dir}
# 指定了alpaca_gpt4_data的话，CUSTOM_DATA_PATH就没用了

CUDA_VISIBLE_DEVICES=${gpu} nohup python main.py $MODEL_PATH ${CUSTOM_DATA_PATH} \
    --wbits $2 \
    --groupsize 16 \
    --perchannel \
    --qq_scale_bits 3 \
    --qq_zero_bits 3 \
    --qq_groupsize 16 \
    --outlier_threshold=0.2 \
    --permutation_order act_order \
    --percdamp 0.01 \
    --nsamples 128 \
    --seed 1234 \
    --seqlen 2048 \
    --skip_out_loss \
    --save ${quantized_model_dir} \
    --pid ${4:--1} > ${quantized_model_dir}/spqr_quant_qwen.log 2>&1 &
echo pid: $!
