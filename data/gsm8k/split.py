import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(file_path, data, mode="w"):
    with open(file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved to {file_path}")

type='origin'

en_data = list(load_jsonl(f'{type}/en_gsm8k.jsonl'))

final_answers = []
new_en_data = []
for item in en_data:
    answer = item['answer'].split('#### ')[1].strip(" ")
    final_answers.append(answer)
    new_data = {
        "id": item['id'],
        "instruction": item['question'],
        "answer": item.get('answer', ""),
        "final_answer": answer,
    }

    new_en_data.append(new_data)

save_jsonl(f'{type}/en_gsm8k.jsonl', new_en_data)
save_jsonl('train/en_gsm8k.jsonl', new_en_data[:7473])
save_jsonl('test/en_gsm8k.jsonl', new_en_data[7473:])

for lang in ['es', 'ru', 'de', 'fr']:
    data = list(load_jsonl(f'{type}/{lang}_gsm8k.jsonl'))
    new_data = []
    for item, answer in zip(data, final_answers):
        d = {
            "id": item['id'],
            "instruction": item['prompt'],
            "answer": item.get('answer', ""),
            "final_answer": answer,
        }
        new_data.append(d)
    save_jsonl(f'{type}/{lang}_gsm8k.jsonl', new_data)
    save_jsonl(f'train/{lang}_gsm8k.jsonl', new_data[:7473])
    save_jsonl(f'test/{lang}_gsm8k.jsonl', new_data[7473:])





