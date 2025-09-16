

conda activate econberta
huggingface-cli login --token $(cat ~/.cache/huggingface/token_write)
CUDA_VISIBLE_DEVICES=0 python train.py