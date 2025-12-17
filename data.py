#conda_env: verl
import os
import json
from datasets import Dataset, load_dataset, Dataset

import argparse

CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)

def chat_template(question):
    prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def custom_load(mode, data_path = "/home/allanz/omega/data/easy_poly_v8.jsonl", turn_off_thinking=False):
    data = []
    with open(data_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            temp_data = json.loads(line)
            print(temp_data)
            prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{temp_data['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            temp_data['prompt'] = prompt
            data.append(temp_data)

    if mode == "train":
        return Dataset.from_list(data).select(range(900))
    if mode == "eval":
        return Dataset.from_list(data).select(range(900, 1000))

if __name__ == '__main__':
    # example code for turing gsm8k into parquet file for verl training

    if not os.path.isdir(os.path.join(PROJECT_ROOT, "datasets")):
        os.makedirs(os.path.join(PROJECT_ROOT, "datasets"))

    dataset_name = "new_poly"
    dataset_save_path = os.path.join(PROJECT_ROOT, "datasets", dataset_name)

    #train_dataset = load_dataset("hiyouga/math12k")["train"]
    #test_dataset = load_dataset("hiyouga/math12k")["test"]
    train_dataset = custom_load("train")
    test_dataset = custom_load("eval")

    print(train_dataset[0])

    # Construct a `def make_map_fn(split)` for the corresponding datasets.
    def make_map_fn(split):
        def process_fn(example, idx):
            #question = chat_template(example["problem"])
            #answer = example["answer"]
            question = example["prompt"]
            # make this a string or else when we grade it won't work

            answer = str(example["ground_truth"])
            data = {
                "data_source": f"{dataset_name}",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(dataset_save_path, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(dataset_save_path, 'test.parquet'))
    print(f"datasets saved to {dataset_save_path}")
