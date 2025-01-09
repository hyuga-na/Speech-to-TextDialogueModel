from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
import sys
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
import torch
import pdb
import os

import pretraining
import organize_data
import finetuning

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser(description="text dialogue model predict response and input's emotion label")
parser.add_argument("-d", "--debug", action="store_true", help="this is flag to use small debug data")
args = parser.parse_args()


def pretrain():
    save_path = "./ModelWeight/Speech2TextModel"
    save_log_path = "./Log/Speech2TextModelPretrain"
    save_model_path = "./ModelWeight/Speech2TextModel"
    pretrained_path = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
    

    # モデル定義
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    
    # 音声トークンの拡張
    tokenizer.add_tokens(list(["speech_"+str(i) for i in range(32000,33000)]))
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    
    if args.debug:
        debug_path = "./Dataset/debug_data.json"
        train_path = val_path = test_path = debug_path
    else:
        train_path = "./Dataset/pretrain_train_data.json"
        val_path = "./Dataset/pretrain_val_data.json"
        test_path = "./Dataset/pretrain_test_data.json"
    

    """build dataset
    List[ Dict{ 
        "input_text": 入力テキスト,
        "output_text": 応答テキスト,
        "emotion": 感情
        "input_ids": tokenize("<s>ユーザー: {入力テキスト}<NL>システム: {応答テキスト}</s>"),
        "attention_mask": [1]*len(input_ids)
        } ]
    """
    train_dataset = organize_data.pretrain_wav(train_path, tokenizer=tokenizer)
    val_dataset = organize_data.pretrain_wav(val_path, tokenizer=tokenizer)
    test_dataset = organize_data.pretrain_wav(test_path, tokenizer=tokenizer)
    print("train_dataset: ",len(train_dataset), end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in train_dataset]))
    print("val_dataset: ",len(val_dataset), end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in val_dataset]))
    print("test_dataset: ",len(test_dataset),end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in test_dataset]))
    print("data size: ", sys.getsizeof(train_dataset))


    pretraining.pretraining(model=model,
                            tokenizer=tokenizer,
                            save_path=save_path,
                            save_model_path=save_model_path,
                            save_log_path=save_log_path,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset)


def pretrain_inference():
    pretrained_path = "./ModelWeight/Speech2TextModel"
    dataset_path = "./Dataset/pretrain_test_data.json"

    # モデル定義
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)

    dataset = organize_data.pretrain_text(dataset_path, tokenizer=tokenizer, inference=True)

    pretraining.inference(model=model,
                          tokenizer=tokenizer,
                          dataset=dataset,
                          ratio=0.1)




"""
モデルconfig
"""
class CustomConfig(PretrainedConfig):
    model_type = "custom_model"
    
    def __init__(self, tokenizer_name="rinna/japanese-gpt-neox-3.6b-instruction-ppo", pretrained_path=None, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_path = pretrained_path
        self.tokenizer_name = tokenizer_name
        self.custom_parameter = kwargs.get("custom_parameter", "default_value")


class Speech2TextDialogueModel(PreTrainedModel):
    config_class = CustomConfig
    def __init__(self, config):
        super().__init__(config)
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_path,
            device_map="auto",
            output_hidden_states=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_path,
            use_fast=False
        )

        self.linear = torch.nn.Linear(self.llm.config.hidden_size, 5)
        self.softmax = torch.nn.Softmax(dim=1)
    

    """応答と[CLS]から感情分類のlogitsを算出
    引数
        input_ids : 入力テキストトークン列
        attention_mask : attention_mask
        emotion : 正解感情
        labels : 正解応答テキスト
    返却値
        outputs = {
            logits : LLMの生成logits
            emotoin_logits : 感情分類のlogits
            response_loss : 応答生成損失
            emotion_loss : 感情予測損失
            その他正解データ
            }
    """
    def forward(self, input_ids, attention_mask=None, emotion=None, labels=None, **kwargs):
        
        input_sequence = torch.cat((input_ids["input_ids"], input_ids["labels"]), dim=-1).to("cuda")
        emotion = input_ids["emotion"]
        attention_mask = torch.full(input_sequence.size(), fill_value=1)
        # 入力部分は無視するために-100で埋める
        labels = torch.cat((torch.full(input_ids["input_ids"].size(),fill_value=-100).to("cuda"), input_ids["labels"].to("cuda")), dim=-1)
        
        # 応答生成
        outputs = self.llm(
            input_ids=input_sequence.reshape(1,-1),
            labels=labels.reshape(1,-1),
            attention_mask=attention_mask.reshape(1,-1))

        # [CLS]トークンの場所を取得
        if self.tokenizer.cls_token_id in input_sequence:
            cls_index = torch.where(input_sequence == self.tokenizer.cls_token_id)
        else:
            cls_index = (0,-1)
        
        # [CLS]トークンの隠れ状態から感情を予測
        hidden_states = outputs.hidden_states[-1]
        emotion_logits = self.softmax(self.linear(hidden_states).squeeze())
        emotion_logits = emotion_logits[cls_index[-1]]
        outputs["emotion_logits"] = emotion_logits
        outputs["emotion"] = emotion
        
        # loss
        dialogue_logits = outputs.logits
        response_loss = torch.nn.CrossEntropyLoss()(dialogue_logits.view(-1, dialogue_logits.size(-1)), labels.view(-1))
        emotion_label = torch.tensor(outputs.emotion).to("cuda")
        emotion_loss = torch.nn.CrossEntropyLoss()(emotion_logits.reshape(1,-1), emotion_label.reshape(-1))
        outputs["response_loss"] = response_loss
        outputs["emotion_loss"] = emotion_loss

        # 学習中の応答と感情の予測結果を確認するためのもの
        response = dialogue_logits.argmax(-1)[0, input_ids["input_ids"].size(-1):]
        response = self.tokenizer.decode(response)
        outputs["predict_response"] = response
        outputs["predict_emotion"] = emotion_logits.argmax(-1)

        return outputs

    """応答生成
    入力
        data = {
            input_ids: 入力トークン列
            input_text: 入力テキスト
            output_text: 正解応答
            emotion: 正解感情
            }
    出力
        results = {
            input_text: 入力テキスト
            output_text: 正解応答テキスト
            emotion: 正解感情
            predict_response: 予測応答
            predict_emotion: 予測感情
        }
    """
    def generate(self, data):
        outputs = self.llm.generate(
            input_ids=data["input_ids"].unsqueeze(0).to("cuda"),
            max_new_tokens=256,
            min_new_tokens=2,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
                
        # [CLS]トークンの隠れ状態から感情を予測
        hidden_states = outputs.hidden_states[-1]
        emotion_logits = self.softmax(self.linear(hidden_states).squeeze())
        emotion_logits = emotion_logits[0] # 0番目に[CLS]があるため
        outputs["emotion_logits"] = emotion_logits
        
        results = {
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "emotion": data["emotion"],
            "predict_response": self.tokenizer.decode(outputs["sequences"][0][data["input_ids"].size(-1):]),#.split("<NL>")[1],
            "predict_emotion": emotion_logits.argmax(-1),
        }
        
        return results


"""
モデルをint8_trainingにセット
"""
def set_model_for_kbit_training(model):
    model = prepare_model_for_kbit_training(model)
    return model

"""
set lora config
"""
def get_lora_config():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config

"""
LoRAセット
"""
def set_model_for_lora_training(model, lora_config):
    model = set_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

"""
print model size
"""
def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))



"""モデルのファインチューニング
"""
def finetune(save_model_path):
    save_path = save_model_path
    save_log_path = "./Log/Speech2TextModelFT"
    pretrained_path = "./ModelWeight/Speech2TextModel"
    

    # モデル定義
    config = CustomConfig(pretrained_path=pretrained_path, tokenizer_name=pretrained_path)
    model = Speech2TextDialogueModel(config)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)


    print(model)
    model.llm = set_model_for_lora_training(model.llm, get_lora_config()) # set lora
    model.linear.weight.requires_grad = False#True
    model.linear.bias.requires_grad = False#True
    print(model.llm.print_trainable_parameters())
    print_model_size(model)

    if args.debug:
        debug_path = "./Dataset/debug_data.json"
        train_path = val_path = test_path = debug_path
    else:
        train_path = "./Dataset/train_data.json"
        val_path = "./Dataset/val_data.json"
        test_path = "./Dataset/test_data.json"
    

    """build dataset
    List[ Dict{ 
        "input_text": 入力テキスト,
        "output_text": 応答テキスト,
        "emotion": 感情
        "input_ids": tokenize("<s>ユーザー: {入力テキスト}<NL>システム: {応答テキスト}</s>"),
        "attention_mask": [1]*len(input_ids)
        } ]
    """
    train_dataset = organize_data.finetune_wav(train_path, tokenizer=tokenizer)
    val_dataset = organize_data.finetune_wav(val_path, tokenizer=tokenizer)
    test_dataset = organize_data.finetune_wav(test_path, tokenizer=tokenizer)
    print("train_dataset: ",len(train_dataset), end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in train_dataset]))
    print("val_dataset: ",len(val_dataset), end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in val_dataset]))
    print("test_dataset: ",len(test_dataset),end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in test_dataset]))
    print("data size: ", sys.getsizeof(train_dataset))


    finetuning.finetune(
        model,
        tokenizer,
        save_path,
        save_model_path,
        save_log_path,
        train_dataset,
        val_dataset,
        test_dataset
        )


"""ファインチューニングしたモデルの推論
"""
def finetune_inference(save_model_path):
    pretrained_path = "./ModelWeight/Speech2TextModel"
    peft_path = save_model_path
    dataset_path = "./Dataset/train_data.json"

    # モデル定義
    config = CustomConfig(pretrained_path=pretrained_path, tokenizer_name=pretrained_path)
    model = Speech2TextDialogueModel(config)
    model.llm = PeftModel.from_pretrained(
        model.llm,
        peft_path,
        device_map="sequence"
    )
    model.linear = torch.load(f"{peft_path}/linear.pth", map_location="cuda")

    print(model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)

    dataset = organize_data.finetune_wav(dataset_path, tokenizer=tokenizer, inference=True)

    finetuning.inference(model=model,
                          tokenizer=tokenizer,
                          dataset=dataset,
                          ratio=0.1)
    

if __name__=="__main__":
    #pretrain()
    #pretrain_inference()

    save_model_path = "./ModelWeight/Speech2TextModelFT_0"
    finetune(save_model_path)
    finetune_inference(save_model_path)
