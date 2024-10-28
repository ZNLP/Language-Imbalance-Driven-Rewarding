#!/bin/bash

# Disable parallelism in tokenizers to avoid conflicts
export TOKENIZERS_PARALLELISM=false

# Define paths and parameters
question_file=${question_file:-"data/x-alpacaeval/data/en.json"}     # Path to question file (default example)
question_key=${question_key:-"instruction"}                          # Key to access questions in JSON file
save_dir=${save_dir:-"output"}                                       # Directory to save generated responses
checkpoint_dir=${checkpoint_dir:-"checkpoints"}                      # Path to model checkpoint
temperature=${temperature:-0}                                        # Sampling temperature (e.g., 0 for deterministic output)
eval_type=${eval_type:-"xalpacaeval"}                                # Evaluation type (e.g., xalpacaeval, mgsm)
lang=${lang:-"en"}                                                   # Language of question file (e.g., "en", "zh")
with_prefix=${with_prefix:-false}                                    # Add prefix to questions (true/false)
template=${template:-"qwen"}                                         # Model template name (e.g., qwen, llama3)
repetition_penalty=${repetition_penalty:-1.1}                        # Repetition penalty for decoding

# Execute inference script with the defined parameters
python -u utils/batch_inference.py \
    --question_file "${question_file}" \
    --question_key "${question_key}" \
    --save_dir "${save_dir}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --temperature "${temperature}" \
    --max_tokens 2048 \
    --seed 42 \
    --eval_type "${eval_type}" \
    --lang "${lang}" \
    --with_prefix "${with_prefix}" \
    --template "${template}" \
    --repetition_penalty "${repetition_penalty}"
