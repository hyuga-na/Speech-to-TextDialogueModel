import json
import torch
from numpy import ndarray
from typing import Dict


"""
トークナイズ
"""
def tokenize(prompt: str, tokenizer, max_length=2048) -> Dict[str, ndarray]:
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {
        "input_ids": torch.tensor(result["input_ids"]),
        "attention_mask": torch.tensor(result["attention_mask"]),
    }


"""
get speech_i ids
"""
def get_wav_ids(features):
    ids = ["speech_"+str(i) for i in features]
    ids = "".join(ids)
    return ids




"""テキスト対話モデルの事前学習用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer : トークナイザー
返却値
    List[ Dict{ 
        "input_ids": tokenize("<s>ユーザー: {音声トークン}[SEP]{書き起こしテキスト}<NL>システム: {応答テキスト}</s>"),
        "attention_mask": [1]*len(input_ids)
        } ]
"""
def pretrain_text(data_path, tokenizer):
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>システム: "+data["output_text"], tokenizer)

        dataset.append(
            ids
        )
    
    return dataset


"""テキスト対話モデルのファインチューニング用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer : トークナイザー
返却値
    List[ Dict{
        "input_text": 入力テキスト
        "output_text": 応答テキスト
        "emotion": 感情
        "input_ids": tokenize("<s>ユーザー: {入力テキスト}<NL><{予測感情}>システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def finetune_text(data_path, tokenizer):
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        input_ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL><"+data["emotion"]+">システム: "+data["output_text"], tokenizer)["input_ids"]
        
        if len(input_ids) > 1000:
            input_ids = input_ids[-1000:]
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "emotion": data["emotion"],
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "attention_mask": torch.tensor(attention_mask).to("cpu"),
            })
    
    return dataset


"""Speech2Text対話モデルの事前学習用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer : トークナイザー
返却値
    List[ Dict{
        "input_text": 入力テキスト
        "output_text": 応答テキスト
        "input_ids": tokenize("<s>ユーザー: {音声}[SEP]{入力テキスト}<NL>システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def pretrain_wav(data_path, tokenizer):    
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        wav_text = get_wav_ids(data["input_wav_ids"])

        input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+data["output_text"], tokenizer)["input_ids"]
        if len(input_ids) > 1000:
            input_ids = input_ids[-1000:]
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "attention_mask": torch.tensor(attention_mask).to("cpu"),
            })
    
    return dataset


"""Speech2Text対話モデルのファインチューニング用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer : トークナイザー
返却値
    List[ Dict{
        "input_text": 入力テキスト
        "output_text": 応答テキスト
        "emotion": 感情
        "input_ids": tokenize("<s>ユーザー: {音声}[SEP]{入力テキスト}<NL><{予測感情}>システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def finetune_wav(data_path, tokenizer):
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        wav_text = get_wav_ids(data["input_wav_ids"])

        input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL><"+data["emotion"]+">システム: "+data["output_text"], tokenizer)["input_ids"]
        if len(input_ids) > 1000:
            input_ids = input_ids[-1000:]
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "emotion": data["emotion"],
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "attention_mask": torch.tensor(attention_mask).to("cpu"),
            })
    
    return dataset
