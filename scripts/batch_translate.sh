#!/bin/bash
#usage: batch_inference.sh <gpu_id> <type> <split>
export TOKENIZERS_PARALLELISM=false

question_file=${question_file:-"data/alpagasus/origin/en.jsonl"}     # Path to inference file (default example)
save_dir=${save_dir:-"output"}                                       # Directory to save generated responses
checkpoint_dir=${checkpoint_dir:-"checkpoints"}                      # Path to model checkpoint
temperature=${temperature:-0}                                        # Sampling temperature (e.g., 0 for deterministic output)
src_lang=${src_lang:-"en"}                                           # Source language, e.g., en, de, fr
tgt_lang=${tgt_lang:-"fr"}                                           # Target language, e.g., en, de, fr
with_prefix=${with_prefix:-false}                                    # Add prefix to questions (true/false)
template=${template:-"qwen"}                                         # Model template name (e.g., qwen, llama3)
repetition_penalty=${repetition_penalty:-1.1}                        # Repetition penalty for decoding

question_file=/public/zhangjiajun/wyang/workspace/release/code/Language-Imbalance-Driven-Rewarding/process_data/inference/Llama-3-8B-Instruct/M0/en.jsonl.prediction.with_Meta-Llama-3-8B-Instruct.temp_0.0_seed_42_repeat_1.1.jsonl
save_dir=/public/zhangjiajun/wyang/workspace/release/code/Language-Imbalance-Driven-Rewarding/process_data/translate/Llama-3-8B-Instruct/M0
checkpoint_dir=/public/zhangjiajun/PretrainModels/Meta-Llama-3-8B-Instruct
temperature=0
src_lang=en
tgt_lang=fr
with_prefix=false
template=llama3

python -u utils/batch_translate.py \
    --question_file ${question_file} \
    --save_dir ${save_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --temperature ${temperature} \
    --max_tokens 4096 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --seed 42 \
    --with_prefix ${with_prefix} \
    --template ${template} 
