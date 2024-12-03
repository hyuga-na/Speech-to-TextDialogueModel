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
"""
import random
def split_data():
    dataset_path = "/mnt/home/hyuga-n/VOICE_DATA/nucc_studies_dataset.json"
    json_open = open(dataset_path, "r")
    dataset_prompts = json.load(json_open)
    print(f"dataset_prompts:{len(dataset_prompts)}, {len(dataset_prompts[0])}")

    random.seed(0)
    random.shuffle(dataset_prompts)
    train_path, val_path, test_path = data_split(dataset_prompts, [7,1,2]) # 訓練とテストデータ分割
    print(f"train: {len(train_path)}, validation: {len(val_path)}, test: {len(test_path)}")

    save_path = "/mnt/home/hyuga-n/VOICE_DATA/nucc_studies_dataset_train.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(train_path, f, ensure_ascii=False, indent=2)

    save_path = "/mnt/home/hyuga-n/VOICE_DATA/nucc_studies_dataset_val.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(val_path, f, ensure_ascii=False, indent=2)

    save_path = "/mnt/home/hyuga-n/VOICE_DATA/nucc_studies_dataset_test.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(test_path, f, ensure_ascii=False, indent=2)


    dataset_path = "/mnt/home/hyuga-n/VOICE_DATA/wav2text_dataset.json"
    json_open = open(dataset_path, "r")
    dataset_prompts = json.load(json_open)
    print(f"dataset_prompts:{len(dataset_prompts)}, {len(dataset_prompts[0])}")

    random.seed(0)
    random.shuffle(dataset_prompts)
    train_path, val_path, test_path = data_split(dataset_prompts, [7,1,2]) # 訓練とテストデータ分割
    print(f"train: {len(train_path)}, validation: {len(val_path)}, test: {len(test_path)}")

    save_path = "/mnt/home/hyuga-n/VOICE_DATA/wav2text_dataset_train.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(train_path, f, ensure_ascii=False, indent=2)

    save_path = "/mnt/home/hyuga-n/VOICE_DATA/wav2text_dataset_val.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(val_path, f, ensure_ascii=False, indent=2)

    save_path = "/mnt/home/hyuga-n/VOICE_DATA/wav2text_dataset_test.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(test_path, f, ensure_ascii=False, indent=2)

def main():
    split_data()

if __name__ == "__main__":
    main()