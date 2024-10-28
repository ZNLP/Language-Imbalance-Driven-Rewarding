import json
import os
from utils import check_repeated_sentences, load_jsonl, write_jsonl

def process_translations(src_lang, tgt_lang, file_path=".", filename = "jsonl.prediction.with_Meta-Llama-3-8B-Instruct.temp_0.0_seed_1234.jsonl", save_path="format/"):
    """
    提取和格式化翻译结果，保存为给定格式的JSONL文件。

    :param src_lang: 源语言标记
    :param tgt_lang: 目标语言标记
    :param base_dir: 基础目录，存放源文件
    :param filename_suffix: 源文件的后缀
    :param format_dir: 格式化后文件的存放目录
    """
    # 构建文件名和路径

    if src_lang != "en":
        raise ValueError("If the source language is not English, the dpo data format should be modifed.")
    
    src_to_tgt_filename = filename.format(src_lang=src_lang, tgt_lang=tgt_lang)
    tgt_to_src_filename = filename.format(src_lang=tgt_lang, tgt_lang=src_lang)

    src_to_tgt_path = f"{file_path}/{src_to_tgt_filename}"
    tgt_to_src_path = f"{file_path}/{tgt_to_src_filename}"

    # # add check_repeated_sentences in English
    # tgt_src_data = load_jsonl(tgt_to_src_path)
    # for item in tgt_src_data:
    #     if check_repeated_sentences(item['response'], threshold=5):
    #         with open("repeated_sentences.txt", "a") as f:
    #             f.write("********repeated Sentences********" + "\n")
    #             f.write(item['response'] + "\n")



    # save_path = f"{save_path}/{src_lang}_and_{tgt_lang}"

    instruction_about_translate = list(load_jsonl(f"process_data/translate/instruction_about_translate.jsonl"))

    instruction_about_translate = [item['id'] for item in instruction_about_translate]


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    src2tgt_data = load_jsonl(src_to_tgt_path)
    tgt2src_data = load_jsonl(tgt_to_src_path)
    src_dpo_data = []
    tgt_dpo_data = []
    src_dpo_file = f"{save_path}/{src_lang}_dpo.jsonl"
    tgt_dpo_file = f"{save_path}/{tgt_lang}_dpo.jsonl"

    count = 0
    for src2tgt_item, tgt2src_item in zip(src2tgt_data, tgt2src_data):
        if src2tgt_item["id"] != tgt2src_item["id"]:
            raise ValueError(f"ID not match: {src2tgt_item['id']} != {tgt2src_item['id']}")
        if tgt2src_item[f"{tgt_lang}2{src_lang}_reponse"]=="Translation Error" or src2tgt_item[f"{src_lang}2{tgt_lang}_reponse"]=="Translation Error":
            continue

        if src2tgt_item["id"] not in instruction_about_translate:
            src_dpo_data.append({
                "prompt": src2tgt_item["prompt"],
                "chosen": src2tgt_item["response"],
                "rejected": tgt2src_item[f"{tgt_lang}2{src_lang}_reponse"],
                "id": src2tgt_item["id"]
            })
            tgt_dpo_data.append({
                "prompt": tgt2src_item["prompt"],
                "chosen": src2tgt_item[f"{src_lang}2{tgt_lang}_reponse"],
                "rejected": tgt2src_item["response"],
                "id": tgt2src_item["id"]
            })
        else:
            count += 1
            src_dpo_data.append({
                "prompt": src2tgt_item["prompt"],
                "chosen": src2tgt_item["response"],
                "rejected": tgt2src_item[f"{tgt_lang}2{src_lang}_reponse"],
                # "rejected": tgt2src_item["response"],
                "id": src2tgt_item["id"]
            })
            tgt_dpo_data.append({
                "prompt": tgt2src_item["prompt"],
                "chosen": src2tgt_item["response"],
                "rejected": tgt2src_item["response"],
                "id": tgt2src_item["id"]
            })

    print(f"count: {count}")
    write_jsonl(src_dpo_file, src_dpo_data)
    write_jsonl(tgt_dpo_file, tgt_dpo_data)

    
# 示例用法
if __name__ == "__main__":

    src_lang = "en"
    tgt_lang = "zh"

    # model="Meta-Llama-3-8B-Instruct"
    model="Aya-23-8B"
    
   
    # file_path = f"/mnt/workspace/huaike.wc/exps/2024-04-17-dual_dpo/dual_dpo_v4/process_data/translate/M0/{model}/subset_id_0_1000/repeat_1.1"
    # save_path = f"/mnt/workspace/huaike.wc/exps/2024-04-17-dual_dpo/dual_dpo_v4/data/alpagasus/M0/{model}/subset_id_0_1000/repeat_1.1"
    

    file_path = f"/mnt/workspace/huaike.wc/exps/2024-04-17-dual_dpo/dual_dpo_v4/process_data/translate/M1/{model}_lr_5e-7_bs_1_ga_4_gpus_4_aya_dpo_apagasus_5k_id_0_1000_repeat_1.1_beta_0.5_ftx_1/subset_id_0_1000/repeat_1.1"
    save_path = f"/mnt/workspace/huaike.wc/exps/2024-04-17-dual_dpo/dual_dpo_v4/data/alpagasus/M1/{model}/subset_id_0_1000/repeat_1.1"


    
    all_langs = ["en", "es", "ru", "de", "fr"]
    for src_lang in ["en"]:
        for tgt_lang in all_langs[1:]:
           
        
            # filename = "{src_lang}.jsonl.prediction." + "with_" + model + ".temp_0.0_seed_42_repeat_1.1.jsonl.to_{tgt_lang}.jsonl"


            filename = "{src_lang}.jsonl.prediction." + "with_" + model + "_lr_5e-7_bs_1_ga_4_gpus_4_aya_dpo_apagasus_5k_id_0_1000_repeat_1.1_beta_0.5_ftx_1.temp_0.0_seed_42_repeat_1.1.jsonl.to_{tgt_lang}.jsonl"
            
            
            process_translations(src_lang, tgt_lang, file_path=file_path, filename=filename, save_path=save_path)


    all_dpo_data = []
    for file in all_langs:
        file = file + "_dpo.jsonl"
        print(f"add {file}...")
        data = list(load_jsonl(f"{save_path}/{file}"))
        all_dpo_data.extend(data)
    write_jsonl(f"{save_path}/en_es_ru_de_fr_dpo.jsonl", all_dpo_data)
    
