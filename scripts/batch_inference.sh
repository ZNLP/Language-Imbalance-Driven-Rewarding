#!/bin/bash

# Disable parallelism in tokenizers to avoid conflicts
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${1:-0}"  # Set GPU ID to use (default 0)

# Define paths and parameters
question_file=${question_file:-"data/alpagasus/origin/en.jsonl"}    # Path to question file (default example)
save_dir=${save_dir:-"output"}                                      # Directory to save generated responses
checkpoint_dir=${checkpoint_dir:-"checkpoints"}                      # Path to model checkpoint
temperature=${temperature:-0}                                        # Sampling temperature (e.g., 0 for deterministic output)
lang=${lang:-"en"}                                                   # Language of question file (e.g., "en", "zh")
with_prefix=${with_prefix:-false}                                    # Add prefix to questions (true/false)
template=${template:-"qwen"}                                         # Model template name (e.g., qwen, llama3)
repetition_penalty=${repetition_penalty:-1.1}                        # Repetition penalty for decoding


### General Instruction following task
question_file=/public/zhangjiajun/wyang/workspace/release/code/Language-Imbalance-Driven-Rewarding/data/alpagasus/subset/id_0_1000/en.jsonl
save_dir=/public/zhangjiajun/wyang/workspace/release/code/Language-Imbalance-Driven-Rewarding/process_data/alpagasus/inference/Llama-3-8B-Instruct/M0
checkpoint_dir=/public/zhangjiajun/PretrainModels/Meta-Llama-3-8B-Instruct
temperature=0
lang=en
with_prefix=false
template=llama3
repetition_penalty=1.1

### Arithmetic reasoning task
question_file=/public/zhangjiajun/wyang/workspace/release/code/Language-Imbalance-Driven-Rewarding/data/gsm8k/train/en.jsonl
save_dir=/public/zhangjiajun/wyang/workspace/release/code/Language-Imbalance-Driven-Rewarding/process_data/gsm8k/inference/Llama-3-8B-Instruct/M0
checkpoint_dir=/public/zhangjiajun/PretrainModels/Meta-Llama-3-8B-Instruct
temperature=0
lang=en
with_prefix=true
template=llama3
repetition_penalty=1.1

# Execute inference script with the defined parameters
python -u utils/batch_inference.py \
    --question_file "${question_file}" \
    --save_dir "${save_dir}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --temperature "${temperature}" \
    --max_tokens 2400 \
    --seed 42 \
    --lang "${lang}" \
    --with_prefix "${with_prefix}" \
    --template "${template}" \
    --repetition_penalty "${repetition_penalty}" \
    --verbose
