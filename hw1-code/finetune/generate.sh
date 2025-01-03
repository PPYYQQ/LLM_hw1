python generate.py \
    --model_name_or_path /home/pku0008/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
    --max_length 512 \
    --trust_remote_code True \
    --use_lora False \
    --lora_dim 32 \
    --lora_scaling 32 \
    --lora_module_name h. \
    --lora_load_path /home/pku0008/homework_1/LLM_hw1/hw1-code/finetune/results/gpt2-alpaca-lora-rank-32-20241225-044441/lora.pt \
    --seed 42 \
    --use_cuda True \
    --output_dir_name gpt2-alpaca-ori-eval
