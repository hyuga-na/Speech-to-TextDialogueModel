import os
import glob
import librosa
import torch
import pdb
import json
import pickle
import nue_asr

"""音響特徴量をクラスタリングするk-means
"""
class KMeans():
    """初期化
    weight_path : 事前学習されたk-meansの重みファイル名
    llm_vocab_size : 語彙を拡張するLLMの元の語彙数(初期値=32000)
    """
    def __init__(self, weight_path, llm_vocab_size=32000):
        self.llm_vocab_size = llm_vocab_size
        with open(weight_path, "rb") as f:
            self.kmeans_model = pickle.load(f)
    
    """クラスタリング
    引数
        feature : 特徴量ベクトル
    返却値
        List[int] : 各特徴量のクラス"""
    def classtering(self, features):
        ids = self.kmeans_model.predict(features.to("cpu"))
        ids = [int(i + self.llm_vocab_size) for i in ids]
        return ids


"""音声からトークンに変換
引数
    wav : 音声データ
返却値
    List[int] : 音声トークン列
"""
kmeans = KMeans(weight_path="/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/kmeans_model.pkl")
asr_model = nue_asr.load_model("rinna/nue-asr").to("cuda")
def speech2ids(wav):
    wav = torch.tensor(wav).unsqueeze(0)
    if wav.size(1) < 4000:
        pad = torch.zeros((1, 4000-wav.size(1)))
        wav = torch.cat((wav, pad), 1)
    
    with torch.no_grad():
        feature = asr_model.encode_audio(wav.half().to("cuda"))
    
    ids = kmeans.classtering(feature[0])
    return ids


""" STUDIESのtxtファイルとwavファイルを統合したjsonファイルを構築する

引数
    root_path : txtファイルがある親ディレクトリパス
    file_path : txtファイルパス
返却値
    List[Dict{
        "emotion": str(感情(relax, angry, happy, sad)),
        "input_text": str(入力テキスト),
        "output_text": str(応答テキスト),
        "input_wav": torch.tensor(1,N)の音声データ}]
"""
def build_json(root_path, file_path):
    with open(file_path, "r") as f:
        data = f.read().splitlines()
    
    prompts = []
    for i in range(len(data)-1):
        role, emotion, input_text = data[i].split("|")[0:3]
        output_text = data[i+1].split("|")[2]

        if emotion == "平静":
            emotion = "relax"
        elif emotion == "怒り":
            emotion = "angry"
        elif emotion == "喜び":
            emotion = "happy"
        elif emotion == "悲しみ":
            emotion = "sad"

        wav_file = os.path.splitext(os.path.basename(file_path))[0]
        if role == "講師":
            wav_file = wav_file + "-Teacher"
        elif role == "男子生徒":
            wav_file = wav_file + "-MStudent"
        elif role == "女子生徒":
            wav_file = wav_file + "-FStudent"
        wav_file = wav_file + "-Turn-" + str(i//2+1).zfill(2) + ".wav"
        
        audio, sr = librosa.load(root_path+"/wav/"+wav_file, sr=None)
        
        ids = speech2ids(audio)
        
        prompts.append({"emotion": emotion,
                        "input_text": input_text,
                        "output_text": output_text,
                        "input_wav_ids": ids})
    return prompts


"""STUDIESを感情考慮Speech-to-Text対話モデル用のデータ形式に変換
引数
    save_path : 構築したデータセットを保存するファイル名
返却値
    None

List[Dict{
    "emotion"
    "input_text"
    "output_text"
    "input_wav"}]
の形式で整形されたSTUDIESをsave_pathにjsonファイルとして保存
"""
def STUDIES(save_path):
    datasets = []
    type_name = ["Long_dialogue", "Short_dialogue"]
    for ls in type_name:
        root_path = "/mnt/home/hyuga-n/VOICE_DATA/STUDIES_voice_data/" + ls
        dataset_path = glob.glob(f"{root_path}/**")

        for path in dataset_path:
            for txt_file in glob.glob(f"{path}/txt/?D??-Dialogue-??.txt"):
                datasets += build_json(root_path=path, file_path=txt_file)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(datasets, f, indent=2)


"""paraling_dataを感情考慮Speech-to-Text対話モデル用のデータ形式に変換
"""
def paraling_data(save_path):
    data_path = "/mnt/home/hyuga-n/E2ESpeechDialogue/S2Tdiscrete/eval_dataset.json"
    old_dataset = open(data_path, "r")
    old_dataset = json.load(old_dataset)

    new_dataset = []
    for data in old_dataset:
        new_dataset.append({
            "no": data["no"],
            "gender": data["gender"],
            "emotion": data["emotion"],
            "input_text": data["input_text"],
            "output_text": data["output_text"],
            "input_wav_ids": data["input_wav_ids"]
        })
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)


def main():
    # build dataset from STUDIES
    STUDIES(save_path="./STUDIES_data.json")
    # build dataset from paraling_data
    paraling_data(save_path="./paraling_data.json")

if __name__ == "__main__":
    main()