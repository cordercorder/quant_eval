#!/bin/bash

# bash evaluate_qwen_chat.sh 7B 1

# bash evaluate_gptq_quant_qwen_chat.sh 7B 8 1

# bash evaluate_gptq_quant_qwen_chat.sh 7B 4 1

# bash evaluate_gptq_quant_qwen_chat.sh 7B 3 1

bash evaluate_gptq_quant_qwen_chat.sh 7B 2 1

# bash evaluate_llm_int8_quant_qwen_chat.sh 7B 1



bash evaluate_qwen_chat.sh 14B 1

bash evaluate_gptq_quant_qwen_chat.sh 14B 8 1

bash evaluate_gptq_quant_qwen_chat.sh 14B 4 1

bash evaluate_gptq_quant_qwen_chat.sh 14B 3 1

bash evaluate_gptq_quant_qwen_chat.sh 14B 2 1

bash evaluate_llm_int8_quant_qwen_chat.sh 14B 1



bash evaluate_qwen_chat.sh 72B 1,2,3

bash evaluate_gptq_quant_qwen_chat.sh 72B 8 1,2

bash evaluate_gptq_quant_qwen_chat.sh 72B 4 1

bash evaluate_gptq_quant_qwen_chat.sh 72B 3 1

bash evaluate_gptq_quant_qwen_chat.sh 72B 2 1

bash evaluate_llm_int8_quant_qwen_chat.sh 72B 1,2



bash evaluate_spqr_quant_qwen_chat.sh 7B w2 1

bash evaluate_spqr_quant_qwen_chat.sh 7B w3 1

bash evaluate_spqr_quant_qwen_chat.sh 7B w4 1

# bash evaluate_spqr_quant_qwen_chat.sh 7B w8 1



bash evaluate_spqr_quant_qwen_chat.sh 14B w2 1

bash evaluate_spqr_quant_qwen_chat.sh 14B w3 1

bash evaluate_spqr_quant_qwen_chat.sh 14B w4 1

bash evaluate_spqr_quant_qwen_chat.sh 14B w8 1



bash evaluate_spqr_quant_qwen_chat.sh 72B w2 5,6,7

bash evaluate_spqr_quant_qwen_chat.sh 72B w3 5,6,7

bash evaluate_spqr_quant_qwen_chat.sh 72B w4 5,6,7

bash evaluate_spqr_quant_qwen_chat.sh 72B w8 5,6,7
