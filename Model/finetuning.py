import json
from numpy import ndarray
import sys
import torch
import transformers
from transformers import  AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from typing import Dict
import pdb
import torch
import argparse
import organize_data
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


parser = argparse.ArgumentParser(description="text dialogue model predict response and input's emotion label")
parser.add_argument("-d", "--debug", action="store_true", help="this is flag to use small debug data")
args = parser.parse_args()


class MyTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        
        labels = inputs["labels"]
        outputs = model.forward(inputs, labels)
        alpha = self.state.global_step / self.state.max_steps
        loss = outputs["response_loss"] + alpha * outputs["emotion_loss"]
        return (loss, outputs) if return_outputs else loss


"""学習進歩を見るために学習中に推論を出力するためのCallback
"""
class PredictionCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, log_step=[1,10,20,40,80,160,320,640,800]):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.log_step = log_step
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step in self.log_step:
            model = kwargs["model"]

            print(f"step: {state.global_step}")
            for feature in self.eval_dataset[:10]:
                input_ids = feature["input_ids"].unsqueeze(0)
                labels = feature["labels"].unsqueeze(0)
                emotion = feature["emotion"]
                attention_mask = feature["attention_mask"].unsqueeze(0)

                with torch.no_grad():
                    outputs = model.generate(feature)
                    """outputs = model.forward(
                        input_ids = input_ids.to("cuda"),
                        labels = labels.to("cuda"),
                        emotion = emotion,
                        attention_mask = attention_mask.to("cuda")
                    )"""
                print(outputs)
                #answer = self.tokenizer.decode(feature["input_ids"])
                #pred = self.tokenizer.decode(outputs["logits"].argmax(-1)[0])
                #print("answer_dialogue: ",answer)
                #print("pred_dialogue: ",pred)
                #print("answer_emotion: ",emotion)
                #print("pred_emotion: ",outputs["emotion_logits"].argmax(-1))
                #print("loss: ",outputs["loss"])
                ##print("emotion_loss: ",outputs["emotion_loss"])
                print()
    
    def on_epoch_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        model = kwargs["model"]
        #pdb.set_trace()

        print(f"epoch: {state.epoch}")
        for feature in self.eval_dataset[:10]:
            input_ids = feature["input_ids"].unsqueeze(0)
            labels = feature["labels"].unsqueeze(0)
            emotion = feature["emotion"]
            attention_mask = feature["attention_mask"].unsqueeze(0)

            with torch.no_grad():
                print(model.generate(feature))
                #outputs = model.forward(
                #    input_ids = input_ids.to("cuda"),
                #    labels = labels.to("cuda"),
                #    emotion = emotion,
                #    attention_mask = attention_mask.to("cuda")
                #)
            
            #answer = self.tokenizer.decode(feature["input_ids"])
            #pred = self.tokenizer.decode(outputs["logits"].argmax(-1)[0])
            #print("answer: ",answer)
            #print("pred: ",pred)
            #print("answer_emotion: ",emotion)
            #print("pred_emotion: ",outputs["emotion_logits"].argmax(-1))
            #print("loss: ",outputs["loss"])
            print()
            


class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):# features: List[Dict{"emotion", "input_ids", "labels", "attention_mask"}]
        
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        emotions = [feature["emotion"] for feature in features]
        
        # padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "emotion": emotions,
            "attention_mask": attention_mask,
        }
    

"""LLMの事前学習
引数
    save_path : 学習結果を保存するパス
    save_model_path : モデルパラメータを保存するパス
    save_log_path : 学習ログを保存するパス
    train_path : 学習データパス
    val_path : 検証データパス
    test_path : テストデータパス

モデルの全てのパラメータを学習し, save_model_pathに保存する
"""
def finetune(model, tokenizer, save_path, save_model_path, save_log_path, train_dataset, val_dataset, test_dataset):
    
    # 学習中の推論結果を見るためのCallback
    prediction_callback = PredictionCallback(eval_dataset=val_dataset[:10], tokenizer=tokenizer)

    # train中のデータ収集方法の構築
    data_collator = DataCollator(tokenizer=tokenizer)

    # learning config
    num_train_epochs = 3
    eval_steps = 1
    save_steps = 100
    logging_steps = 10
    save_total_limit = 3
    trainer = MyTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            logging_steps=logging_steps,
            logging_dir=save_log_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            #eval_steps=eval_steps,
            #save_steps=save_steps,
            output_dir=save_path,
            report_to="none",
            save_total_limit=save_total_limit,
            push_to_hub=False,
            auto_find_batch_size=False,
            load_best_model_at_end=True,
            label_names=["labels","emotion"],
            fp16=True,
            dataloader_num_workers=16,
        ),
        data_collator=data_collator,
        callbacks=[prediction_callback],
    )
    
    model.config.use_cache=False
    trainer.train()
    model.config.use_cache=True

    # save model & tokenizer
    trainer.save_state()
    tokenizer.save_pretrained(save_model_path)
    trainer.model.llm.save_pretrained(save_model_path)
    torch.save(model.linear, f"{save_model_path}/linear.pth")


    model.eval()
    pred_result = trainer.evaluate(test_dataset, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])
    
    print("test_loss: ", pred_result["eval_loss"])
    


"""テキストモデルの推論
引数
    model : 推論するモデル
    tokenizer : 推論するトークナイザー
    dataset : 推論するデータセット
    ratio=0.1 : 生成する頻度 default=10%
"""
import random
def inference(model, tokenizer, dataset, ratio=0.1):
    print(model)
    model.eval()

    random.seed(0)
    
    for data in dataset[:100]:
        if random.random() < ratio:
            with torch.no_grad():
                outputs = model.generate(data)
            print(outputs)
            print()
            continue
            """outputs = model.llm.generate(
                        input_ids=data["input_ids"].unsqueeze(0).to("cuda"),
                        max_new_tokens=256,
                        min_new_tokens=2,
                        do_sample=True,
                        temperature=0.8,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )
                
                cls_index = torch.where(outputs["sequences"] == model.tokenizer.cls_token_id)
                #pdb.set_trace()
                hidden_state = [tensor[cls_index] for tensor in outputs["hidden_states"][0]]
                hidden_state = sum(hidden_state)
                #hidden_state = outputs["hidden_states"][0][:][0][0]
                
                hidden_state = outputs["hidden_states"][0][-1][0][0] # (len(generated_sequences), attention_layers=37, batch_size, past_hidden_state)
                emotion_logits = model.linear(hidden_state.to("cuda"))
            
            print("answer: ",data["input_text"]+"<NL>システム: "+data["output_text"])
            print("pred: ",tokenizer.decode(outputs["sequences"][0]).split("<NL>")[1])
            print("answer_emotion: ", data["emotion"])
            print("pred_emotion: ",emotion_logits.argmax(-1))
            #pdb.set_trace()
            print()"""
