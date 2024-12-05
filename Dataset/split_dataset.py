import os
import tqdm
import json
import glob
import pdb


"""訓練，検証，テストデータ分割
引数
    prompts : List[str] データセット
    split_ratio : List 訓練，検証，テストの割合
返却値
    train_prompts, val_prompts, test_prompts : List
"""
from typing import List, Dict, Tuple
def data_split(prompts: List[str], split_ratio: List) -> Tuple[List[str]]:
    split_ratio = [i / sum(split_ratio) for i in split_ratio]
    train_len = int(len(prompts)*split_ratio[0])
    test_len = int(len(prompts)*split_ratio[2])
    train_prompts, val_prompts, test_prompts = prompts[:train_len], prompts[train_len:-test_len], prompts[-test_len:]
    return train_prompts, val_prompts, test_prompts


"""データの分割
引数
    dataset_path : 分割するデータセットパス
返却値
    train_data, val_data, test_data : 分割後のデータ
"""
import random
def split_data(dataset_path, split_ratio: List):
    json_open = open(dataset_path, "r")
    dataset_prompts = json.load(json_open)
    print(f"dataset_prompts:{len(dataset_prompts)}, {len(dataset_prompts[0])}")

    random.seed(0)
    random.shuffle(dataset_prompts)
    train_data, val_data, test_data = data_split(dataset_prompts, split_ratio) # 訓練とテストデータ分割
    print(f"train: {len(train_data)}, validation: {len(val_data)}, test: {len(test_data)}")
    
    return train_data, val_data, test_data


def split_paraling_data():
    json_open = open("./Dataset/paraling_data.json", "r")
    dataset_prompts = json.load(json_open)

    for i in range(len(dataset_prompts)):
        dataset_prompts[i]["output_text"] = dataset_prompts[i]["output_text"][0]

    eval_data = dataset_prompts[:100] # evaluate data for paralinguistic

    train_data = [data for data in dataset_prompts[100:] if data["output_text"]]
    
    return train_data, eval_data

"""paraling_dataの10発話全てに応答が付与されているものを確認
"""
def check():
    json_open = open("./Dataset/paraling_data.json", "r")
    dataset_prompts = json.load(json_open)

    count = 0
    num = 0
    no = 0
    for data in dataset_prompts:
        prompt = data["output_text"][0]
        count += 1
        if bool(prompt):
            no += 1
            if no == 10:
                print(data["no"])
                no = 0
                num += 1
        else:
            no = 0
        
        if count == 10:
            count = 0
            no = 0
    
    print("num=",num)


def main():
    paraling_train, paraling_eval = split_paraling_data()

    dataset_path = "./Dataset/STUDIES_data.json"
    train, val, test = split_data(dataset_path, split_ratio=[7,1,2])

    train += paraling_train
    with open("./Dataset/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    
    with open("./Dataset/val_data.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    with open("./Dataset/test_data.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
    
    with open("./Dataset/evaluate_data.json", "w", encoding="utf-8") as f:
        json.dump(paraling_eval, f, ensure_ascii=False, indent=2)
    
    with open("./Dataset/debug_data.json", "w", encoding="utf-8") as f:
        json.dump(train[:100], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    #check()
    main()