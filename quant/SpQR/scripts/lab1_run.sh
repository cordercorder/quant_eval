cd /home/work/jinrenren/quant_eval/quant/SpQR
bash lab1.sh 1 2 14B
echo 11111111111111111111111111111
cd /home/work/jinrenren/quant_eval/benchmarks/ppl/scripts
CUDA_VISIBLE_DEVICES=1 bash evaluate_spqr_quant_qwen_chat_lab1.sh 14B w2
echo 22222222222222222222222222222



cd /home/work/jinrenren/quant_eval/quant/SpQR
bash lab1.sh 1 2 72B
echo 33333333333333333333333333333

cd /home/work/jinrenren/quant_eval/benchmarks/ppl/scripts
CUDA_VISIBLE_DEVICES=1,2 bash evaluate_spqr_quant_qwen_chat_lab1.sh 72B w2
echo 44444444444444444444444444444


# lab2
cd /home/work/jinrenren/quant_eval/quant/SpQR
bash lab2.sh 1 2 14B
echo 555555555555555555555555555555
cd /home/work/jinrenren/quant_eval/benchmarks/ppl/scripts
CUDA_VISIBLE_DEVICES=1 bash evaluate_spqr_quant_qwen_chat_lab2.sh 14B w2
echo 666666666666666666666666666666



cd /home/work/jinrenren/quant_eval/quant/SpQR
bash lab2.sh 1 2 72B
echo 777777777777777777777777777777

cd /home/work/jinrenren/quant_eval/benchmarks/ppl/scripts
CUDA_VISIBLE_DEVICES=1,2 bash evaluate_spqr_quant_qwen_chat_lab2.sh 72B w2
echo 88888888888888888888888888888

