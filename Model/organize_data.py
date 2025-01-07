import json
import torch
from numpy import ndarray
from typing import Dict
from transformers import  AutoTokenizer

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


def emotion_to_id(emotion):
    e2i = {
        "happy": 0,
        "relax": 1,
        "angry": 2,
        "sad": 3,
        "neutral": 4,
    }
    return e2i[emotion]


"""テキスト対話モデルの事前学習用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer : トークナイザー
返却値
    List[ Dict{ 
        "input_ids": tokenize("<s>ユーザー: {入力テキスト}<NL>[CLS]システム: {応答テキスト}</s>"),
        "attention_mask": [1]*len(input_ids)
        } ]
"""
def pretrain_text(data_path, tokenizer, inference=False):
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    
    if inference:
        for data in data_path:
            ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>", tokenizer)
            dataset.append({
                "input_text": data["input_text"],
                "output_text": data["output_text"],
                "input_ids": ids["input_ids"][:-1],
                "attention_mask": ids["attention_mask"][:-1],
            })
    else:
        for data in data_path:
            ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>[CLS]システム: "+data["output_text"], tokenizer)
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
        "emotion": 感情ラベル
        "input_ids": tokenize("<s>ユーザー: {入力テキスト}<NL>[CLS]システム: {応答テキスト}),
        "labels": input_ids 対話学習用ラベル
        "attention_mask": [1]*len(input_ids)
    }]
"""
def finetune_text(data_path, tokenizer, inference=False):
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        if inference: # 推論
            input_ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>", tokenizer)["input_ids"][:-1]
            labels = tokenize("[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]
        else: # 学習
            input_ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]
            labels = input_ids
        
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "emotion": emotion_to_id(data["emotion"]),
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "labels": torch.tensor(labels).to("cpu"),
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
        "input_ids": tokenize("<s>ユーザー: {音声}[SEP]{入力テキスト}<NL>[CLS]システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def pretrain_wav(data_path, tokenizer, inference=False):   
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        wav_text = get_wav_ids(data["input_wav_ids"])

        if inference:
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>", tokenizer)["input_ids"][:-1]
        else:
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]

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
        "input_ids": tokenize("<s>ユーザー: {音声}[SEP]{入力テキスト}<NL>[CLS]システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def finetune_wav(data_path, tokenizer, inference=False):
    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        wav_text = get_wav_ids(data["input_wav_ids"])

        #input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>", tokenizer)["input_ids"][:-1]
        #labels = tokenize("[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]

        if inference:
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>", tokenizer)["input_ids"][:-1]
            labels = tokenize("[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]
        else:
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]
            labels = input_ids

        if len(input_ids) > 1000:
            input_ids = input_ids[-1000:]
            labels = labels[-1000:]
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "emotion": emotion_to_id(data["emotion"]),
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "labels": torch.tensor(labels).to("cpu"),
            "attention_mask": torch.tensor(attention_mask).to("cpu"),
            })
    
    return dataset



"""テキスト対話モデルの評価用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer_path : トークナイザーのパス
返却値
    List[ Dict{
        "no": データ番号
        "input_text": 入力テキスト
        "output_text": 応答テキスト
        "gender": 性別
        "emotion": 感情
        "input_ids": tokenize("<s>ユーザー: {入力テキスト}<NL>[CLS]システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def eval_text(data_path, tokenizer_path, inference=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        if inference:
            input_ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>", tokenizer)["input_ids"][:-1]
        else:
            input_ids = tokenize("<s>ユーザー: "+data["input_text"]+"<NL>[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]
            
        if len(input_ids) > 1000:
            input_ids = input_ids[-1000:]
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "no": data["no"],
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "gender": data["gender"],
            "emotion": data["emotion"],
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "attention_mask": torch.tensor(attention_mask).to("cpu"),
            })
    
    return dataset


"""Speech2Text対話モデルの評価用にデータセットを編集
引数
    data_path : 編集するデータセットjsonファイル
    tokenizer_path : トークナイザーのパス
返却値
    List[ Dict{
        "no": データ番号
        "input_text": 入力テキスト
        "output_text": 応答テキスト
        "gender": 性別
        "emotion": 感情
        "input_ids": tokenize("<s>ユーザー: {音声}[SEP]{入力テキスト}<NL>[CLS]システム: {応答テキスト}),
        "attention_mask": [1]*len(input_ids)
    }]
"""
def eval_wav(data_path, tokenizer_path, inference=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    dataset = []
    data_path = open(data_path, "r")
    data_path = json.load(data_path)
    for data in data_path:
        wav_text = get_wav_ids(data["input_wav_ids"])

        if inference:
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>", tokenizer)["input_ids"][:-1]
        else:
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>[CLS]システム: "+data["output_text"], tokenizer)["input_ids"]

        if len(input_ids) > 1000:
            input_ids = input_ids[-1000:]
        attention_mask = [1] * len(input_ids)
        
        dataset.append({
            "no": data["no"],
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "gender": data["gender"],
            "emotion": data["emotion"],
            "input_ids": torch.tensor(input_ids).to("cpu"),
            "attention_mask": torch.tensor(attention_mask).to("cpu"),
            })
    
    return dataset
