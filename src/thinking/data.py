import json
from datasets import load_dataset, Dataset


def chat_template(question, turn_off_thinking=False):
    prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
    if turn_off_thinking:
        prompt += f"<|im_start|>user\n{question}/no_think<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def load_calc_mixed(mode: str, turn_off_thinking=False):
    if mode == "train":
        dataset = load_dataset("sunyiyou/math_arithmetic_mixed_7B_train")["train"].select(range(900))
    elif mode == "eval":
        dataset = load_dataset("sunyiyou/math_arithmetic_mixed_7B_train")["train"].select(range(900, 1000))
    
    reformatted = []
    for element in dataset:
        prompt = element["messages"][0]["content"].split("\n\n")[0]
        new_data = {"prompt": chat_template(prompt), "ground_truth": element["ground_truth"]}
        reformatted.append(new_data)

    return Dataset.from_list(reformatted)

def load_new_gcd(mode, data_path="/home/allanz/omega/data/easy_gcd.jsonl"):
    new_data = []
    with open(data_path, "r") as file:
        for line in file:
            temp_data = json.loads(line)
            temp_data["prompt"] = chat_template(temp_data["prompt"])
            new_data.append(temp_data)
    if mode == "train":
        return Dataset.from_list(new_data[:900])
    elif mode == "eval":
        return Dataset.from_list(new_data[900:1000])


def load_new_poly(mode, data_path="/home/allanz/omega/data/easy_poly_v8.jsonl"):
    new_data = []
    with open(data_path, "r") as file:
        for line in file:
            temp_data = json.loads(line)
            temp_data["prompt"] = chat_template(temp_data["prompt"])
            new_data.append(temp_data)
    if mode == "train":
        return Dataset.from_list(new_data[:900])
    elif mode == "eval":
        return Dataset.from_list(new_data[900:1000])

if __name__ == "__main__":
    test = load_calc_mixed("train")
    print(test[0])
