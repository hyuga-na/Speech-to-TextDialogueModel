import json
from numpy import ndarray
import sys
import torch
import transformers
from transformers import  AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from typing import Dict
import pdb
import torch

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

"""
get speech_i ids
"""
def get_wav_ids(features):
    ids = ["speech_"+str(i) for i in features]
    ids = "".join(ids)
    return ids

"""
tuning pretrained llm to asr task
"""
def dialogue_lora(save_path, save_model_path, pretrained_path, save_log_path, train_path, val_path, test_path, debug_path, is_debug):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        #load_in_8bit=True,
        device_map="auto",
        #device_map="sequential",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)

    print(model)
    model = set_model_for_lora_training(model, get_lora_config()) # set lora
    print(model.print_trainable_parameters())
    print_model_size(model)

    """
    build dataset
    List[ Dict{ "input_ids": tokenize("<s>ユーザー: {音声トークン}[SEP]{書き起こしテキスト}<NL>システム: {応答テキスト}</s>"), "attention_mask": [1]*len(input_ids)} ]
    """
    if is_debug:
        print("build debug dataset")
        debug_dataset = []
        data_path = open(debug_path, "r")
        data_path = json.load(data_path)
        for data in data_path:
            # prompt

            wav_text = get_wav_ids(data["input_wav_ids"])

            # ids = Dict{"input_ids": torch.tensor(tokens), "attention_mask": torch.tensor([1]*len(tokens))}
            ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+data["output_text"], tokenizer)

            debug_dataset.append(
                ids
            )
        # paraling-eval
        data_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/eval_dataset.json"
        dataset_paraling_eval = open(data_path, "r")
        dataset_paraling_eval = json.load(dataset_paraling_eval)[:-1000]
        for data in dataset_paraling_eval:
            wav_text = get_wav_ids(data["input_wav_ids"])
            if len(data["output_text"][0]) == 0:
                response = "すみません、聞き取れませんでした。"
            else:
                response = data["output_text"][0]
            ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+response, tokenizer)
            debug_dataset.append(
                ids
            )
        #pdb.set_trace()

        train_dataset = debug_dataset
        val_dataset = debug_dataset
        test_dataset = debug_dataset
    else:
        train_dataset = []
        data_path = open(train_path, "r")
        data_path = json.load(data_path)
        for data in data_path:
            wav_text = get_wav_ids(data["input_wav_ids"])

            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+data["output_text"], tokenizer)["input_ids"]
            
            if len(input_ids) > 1000:
                input_ids = input_ids[-1000:]
            attention_mask = [1] * len(input_ids)
            
            train_dataset.append({
                "input_text": data["input_text"],
                "input_ids": torch.tensor(input_ids).to("cpu"),
                "attention_mask": torch.tensor(attention_mask).to("cpu"),
                })
        # paraling-eval
        data_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/eval_dataset.json"
        dataset_paraling_eval = open(data_path, "r")
        dataset_paraling_eval = json.load(dataset_paraling_eval)[:-1000]
        for data in dataset_paraling_eval:
            wav_text = get_wav_ids(data["input_wav_ids"])
            if len(data["output_text"][0]) == 0:
                response = "すみません、聞き取れませんでした。"
            else:
                response = data["output_text"][0]
            ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+response, tokenizer)
            train_dataset.append(
                ids
            )
        
        
        val_dataset = []
        data_path = open(val_path, "r")
        data_path = json.load(data_path)
        for data in data_path:
            wav_text = get_wav_ids(data["input_wav_ids"])

            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+data["output_text"], tokenizer)["input_ids"]
            if len(input_ids) > 1000:
                input_ids = input_ids[-1000:]
            attention_mask = [1] * len(input_ids)
            
            val_dataset.append({
                "input_text": data["input_text"],
                "input_ids": torch.tensor(input_ids).to("cpu"),
                "attention_mask": torch.tensor(attention_mask).to("cpu"),
                })

        test_dataset = []
        data_path = open(test_path, "r")
        data_path = json.load(data_path)
        for data in data_path:
            wav_text = get_wav_ids(data["input_wav_ids"])

            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"]+"<NL>システム: "+data["output_text"], tokenizer)["input_ids"]
            if len(input_ids) > 1000:
                input_ids = input_ids[-1000:]
            attention_mask = [1] * len(input_ids)
            
            test_dataset.append({
                "input_text": data["input_text"],
                "input_ids": torch.tensor(input_ids).to("cpu"),
                "attention_mask": torch.tensor(attention_mask).to("cpu"),
                })
    print("train_dataset: ",len(train_dataset), end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in train_dataset]))
    print("val_dataset: ",len(val_dataset), end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in val_dataset]))
    print("test_dataset: ",len(test_dataset),end=", ")
    print("max_input_ids_size: ",max([len(data["input_ids"]) for data in test_dataset]))
    print("data size: ", sys.getsizeof(train_dataset))

    del data_path
    torch.cuda.empty_cache()

    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
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

                    with torch.no_grad():
                        outputs = model.forward(
                            input_ids=input_ids.to("cuda"),
                            labels=input_ids.to("cuda")
                        )
                    
                    answer = self.tokenizer.decode(feature["input_ids"])
                    pred = self.tokenizer.decode(outputs["logits"].argmax(-1)[0])
                    print("answer: ",answer)
                    print("pred: ",pred)
                    print("loss: ",outputs["loss"])
                    print()
        
        def on_epoch_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            model = kwargs["model"]
            #pdb.set_trace()

            print(f"epoch: {state.epoch}")
            for feature in self.eval_dataset[:10]:
                input_ids = feature["input_ids"].unsqueeze(0)

                with torch.no_grad():
                    outputs = model.forward(
                        input_ids=input_ids.to("cuda"),
                        labels=input_ids.to("cuda")
                    )
                
                answer = self.tokenizer.decode(feature["input_ids"])
                pred = self.tokenizer.decode(outputs["logits"].argmax(-1)[0])
                print("answer: ",answer)
                print("pred: ",pred)
                print("loss: ",outputs["loss"])
                print()
    
    prediction_callback = PredictionCallback(eval_dataset=val_dataset[:10], tokenizer=tokenizer)

    num_train_epochs = 3
    eval_steps = 1
    save_steps = 100
    logging_steps = 10
    save_total_limit = 3
    trainer = transformers.Trainer(
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
            label_names=["labels"],
            fp16=True,
            dataloader_num_workers=4,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
        callbacks=[prediction_callback]
    )
    
    model.config.use_cache=False
    trainer.train()
    model.config.use_cache=True

    # save model & tokenizer
    trainer.save_state()
    tokenizer.save_pretrained(save_model_path)
    trainer.save_model(save_model_path)

    model.eval()
    pred_result = trainer.evaluate(test_dataset, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])
    
    print("test_loss: ", pred_result["eval_loss"])
    
"""
generate sentence using trained model
"""
import random
def inference(pretrained_path, peft_name):
    print("inference")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
        
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        device_map="sequential",
        )
    model = PeftModel.from_pretrained(
        model,
        peft_name,
        device_map="sequence"
    )
    print(model)
    model.eval()

    random.seed(0)

    for data_type in ["debug","train", "val", "test"]:
        print(f"build dataset for {data_type}")
        path = f"/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/dataset_{data_type}.json"
        
        data_path = open(path, "r")
        data_path = json.load(data_path)
        
        for data in data_path[:10]:
            # prompt
            wav_text = get_wav_ids(data["input_wav_ids"])
            input_ids = tokenize("<s>ユーザー: "+wav_text+"[SEP]"+data["input_text"], tokenizer)["input_ids"]
            input_ids = input_ids[:-1].unsqueeze(0) # cut </s>         
            
            if random.random() < 1:
                with torch.no_grad():
                    outputs = model.generate(
                            input_ids.to("cuda"),
                            max_new_tokens=256,
                            min_new_tokens=2,
                            do_sample=True,
                            temperature=0.8,
                            repetition_penalty=1.2,
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                print("answer: ",data["input_text"]+"<NL>システム: "+data["output_text"])
                print("pred: ",tokenizer.decode(outputs[0]).split("[SEP]")[1])
                print()


if __name__ == '__main__':
    is_debug = False # If True use debug dataset 100 size, False use train, eval and test dataset
    if is_debug:
        save_path = "./Speech-to-TextDialogue/Speech-to-TextDialogueModel/ModelWeight/TextBaseModel"#_debug"
        save_model_path = "./Speech-to-TextDialogue/Speech-to-TextDialogueModel/ModelWeight/TextBaseModel"#_debug"
        pretrained_path = ""
    else:
        save_path = ""
        save_model_path = ""
        pretrained_path = ""

    save_log_path = f"{save_path}/speech2text.log"
    train_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/dataset_train.json"
    val_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/dataset_val.json"
    test_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/dataset_test.json"
    debug_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/dataset_debug.json"
    dialogue_lora(
        save_path = save_path,
        save_model_path=save_model_path,
        pretrained_path=pretrained_path,
        save_log_path = save_log_path,
        train_path = train_path,
        val_path = val_path,
        test_path = test_path,
        debug_path = debug_path,
        is_debug = is_debug, 
    )
    inference(
        pretrained_path=save_model_path,
        peft_name=save_model_path
    )