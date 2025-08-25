#!/bin/bash
# ================================================
# Script for multilingual AlpacaEval evaluations
# Author: zhangjiajun
# ================================================

# ---- API Key ----
export OPENAI_API_KEY="xxx"

# ---- Languages to evaluate ----
# Uncomment the types you need
types=("en" "es" "ru" "de" "fr")

# ---- Models to evaluate ----
# Keep only the ones you want active, comment others
# Example sets:
# models=("Meta-Llama-3-8B-Instruct" "MDPO_Meta-Llama-3-8B-Instruct_M0" "MDPO_Meta-Llama-3-8B-Instruct_M1")
# models=("Qwen2-7B-Instruct" "MDPO_Qwen2-7B-Instruct_M0" "MDPO_Qwen2-7B-Instruct_M1")
# models=("zephyr-7b-beta" "zephyr-7b-beta-align-with-DICE")
# models=("Llama-3-Base-8B-SFT" "Llama-3-Base-8B-SFT-DPO")

# Active model set (example):
models=("ICR_M0_Llama-3-Base-8B-SFT-DPO_en_es_ru_de_fr" "ICR_M1_Llama-3-Base-8B-SFT-DPO_en_es_ru_de_fr")

# ---- Paths ----
code_dir=/public/zhangjiajun/wyang/workspace/release/code/Implicit-Cross-Lingual-Rewarding
alpaca_eval_ens=/public/zhangjiajun/anaconda3/envs/alpaca2/bin/alpaca_eval

# ================================================
# Evaluate with GPT-4-Turbo as annotator
# ================================================
for t in "${types[@]}"; do
    for model in "${models[@]}"; do
        ${alpaca_eval_ens} evaluate \
            --model_outputs ${code_dir}/results/base_model/x-alpacaeval_format/${t}/${model}/model_outputs.json \
            --reference_outputs ${code_dir}/results/base_model/x-alpacaeval_format/${t}/gpt-4-turbo-2024-04-09/model_outputs.json \
            --annotators_config weighted_alpaca_eval_gpt4_turbo_multilingual \
            --precomputed_leaderboard ${code_dir}/results/base_model/x_alpacaeval_leaderboard/${t}.csv
    done
done

# ================================================
# (Optional) Evaluate with GPT-4o as annotator
# Uncomment if needed
# ================================================
# for t in "${types[@]}"; do
#     for model in "${models[@]}"; do
#         ${alpaca_eval_ens} evaluate \
#             --model_outputs ${code_dir}/results/base_model/x-alpacaeval_format/${t}/${model}/model_outputs.json \
#             --reference_outputs ${code_dir}/results/base_model/x-alpacaeval_format/${t}/gpt-4-turbo-2024-04-09/model_outputs.json \
#             --annotators_config weighted_alpaca_eval_gpt4o_multilingual \
#             --precomputed_leaderboard ${code_dir}/results/base_model/x_alpacaeval_leaderboard_with_gpt_4o/${t}.csv
#     done
# done

# ================================================
# (Optional) Make a leaderboard
# Example usage
# ================================================
# ${alpaca_eval_ens} make_leaderboard \
#   --leaderboard_path "${code_dir}/results/base_model/leaderboard/multilingual_alpacaeval.csv"  \
#   --all_model_outputs "${code_dir}/results/base_model/x-alpacaeval_subset_format/en/Aya-23-8B/model_outputs.json"  \
#   --reference_outputs "${code_dir}/results/base_model/x-alpacaeval_subset_format/en/gpt-4-turbo/model_outputs.json" \
#   --annotators_config "${code_dir}/alpaca_eval/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo_multilingual/configs.yaml"
