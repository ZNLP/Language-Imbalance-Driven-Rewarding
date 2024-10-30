import os
import json
import re
from termcolor import colored

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                print(colored(f"Exception: {line}", "red"))
    return data

def load_json(file_path):
    data = []
    with open(file_path, "r") as f:
        try:
            data = json.load(f) 
        except json.JSONDecodeError as e:
            print(colored(f"JSON Decode Exception when reading the file: {file_path}\nError: {e}", "red"))
        except Exception as e:
            print(colored(f"Unexpected Exception when reading the file: {file_path}\nError: {e}", "red"))
    return data


def write_jsonl(file_path, data, mode="w"):
    with open(file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(colored(f"Saved to {file_path}", "green"))

def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(colored(f"Saved to {file_path}", "green"))

def write_jsonl_single_data(sample, save_path, mode="a"):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")



def get_file_names(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


from collections import Counter
def check_repeated_sentences(paragraph, threshold=5):
    paragraph = paragraph.replace('\n\n','\n')
    sentences = paragraph.split('\n')
    sentence_counts = Counter(sentences)
    
    for sentence, count in sentence_counts.items():
        if count >= threshold:
            return True
    return False

def custom_sort_key(item):
    """Extract numeric part and prefix from the id for sorting."""
    id = item['id']
    # Assuming the format is always 'prefix-numeric'
    prefix, numeric_part = id.split('-')
    return (int(numeric_part), prefix)  # Note the order: number before prefix
