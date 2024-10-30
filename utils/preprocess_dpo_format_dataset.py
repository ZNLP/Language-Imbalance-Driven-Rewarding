import os
from typing import List
from utils import check_repeated_sentences, load_jsonl, write_jsonl

def process_translations(
    src_lang: str, 
    tgt_lang: str, 
    file_path: str = ".", 
    filename: str = "translation.jsonl", 
    save_path: str = "formatted/"
):
    """
    Extract and format translation results, saving them as JSONL files.
    
    :param src_lang: Source language code
    :param tgt_lang: Target language code
    :param file_path: Directory path containing the source files
    :param filename: Template filename for the source file
    :param save_path: Directory path for saving the formatted files
    """
    if src_lang != "en":
        raise ValueError("Source language should be English for this format.")
    
    # Construct file paths for source-to-target and target-to-source translations
    src_to_tgt_path = os.path.join(file_path, filename.format(src_lang=src_lang, tgt_lang=tgt_lang))
    tgt_to_src_path = os.path.join(file_path, filename.format(src_lang=tgt_lang, tgt_lang=src_lang))
    
    # Load translation instruction data
    instruction_about_translate = [item['id'] for item in load_jsonl("instructions.jsonl")]

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load source-to-target and target-to-source translation data
    src2tgt_data = load_jsonl(src_to_tgt_path)
    tgt2src_data = load_jsonl(tgt_to_src_path)
    src_dpo_data, tgt_dpo_data = [], []
    
    for src2tgt_item, tgt2src_item in zip(src2tgt_data, tgt2src_data):
        if src2tgt_item["id"] != tgt2src_item["id"]:
            raise ValueError(f"ID mismatch: {src2tgt_item['id']} != {tgt2src_item['id']}")

        # Skip entries marked as "Translation Error"
        if tgt2src_item[f"{tgt_lang}2{src_lang}_reponse"] == "Translation Error" or src2tgt_item[f"{src_lang}2{tgt_lang}_reponse"] == "Translation Error":
            continue

        # Build DPO data based on whether it has special instructions
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
            src_dpo_data.append({
                "prompt": src2tgt_item["prompt"],
                "chosen": src2tgt_item["response"],
                "rejected": tgt2src_item[f"{tgt_lang}2{src_lang}_reponse"],
                "id": src2tgt_item["id"]
            })
            tgt_dpo_data.append({
                "prompt": tgt2src_item["prompt"],
                "chosen": src2tgt_item["response"],
                "rejected": tgt2src_item["response"],
                "id": tgt2src_item["id"]
            })

    # Save formatted DPO data to JSONL files
    write_jsonl(os.path.join(save_path, f"{src_lang}_dpo.jsonl"), src_dpo_data)
    write_jsonl(os.path.join(save_path, f"{tgt_lang}_dpo.jsonl"), tgt_dpo_data)

def aggregate_dpo_data(save_path: str, all_langs: List[str]):
    """
    Aggregate all DPO data files into a single JSONL file.
    
    :param save_path: Path where data files are saved
    :param all_langs: List of all languages
    """
    all_dpo_data = []
    for lang in all_langs:
        file_path = os.path.join(save_path, f"{lang}_dpo.jsonl")
        print(f"Adding {file_path}...")
        data = list(load_jsonl(file_path))
        all_dpo_data.extend(data)
    
    write_jsonl(os.path.join(save_path, "all_languages_dpo.jsonl"), all_dpo_data)

# Example usage
if __name__ == "__main__":
    src_lang = "en"
    tgt_langs = ["es", "ru", "de", "fr"]  # Example target languages
    file_path = "./data"
    save_path = "./formatted_data"

    # Process translation data for each language pair
    for tgt_lang in tgt_langs:
        process_translations(src_lang=src_lang, tgt_lang=tgt_lang, file_path=file_path, save_path=save_path)

    # Aggregate all DPO data files
    aggregate_dpo_data(save_path=save_path, all_langs=[src_lang] + tgt_langs)
