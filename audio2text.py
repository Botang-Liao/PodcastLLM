import whisper
from tqdm import tqdm
import os
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# 加載 Whisper 模型
model = whisper.load_model("medium") #whisper_model{'tiny','base,'small','medium','large-v2/v3'}

def transcribe_audio(file_path, output_file):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # 進行音訊轉錄
    result = model.transcribe(file_path, fp16=device)


    # 直接寫入轉錄結果
    with open(output_file, "w", encoding="utf-8") as f:
        segments = result["segments"]
        current_text = ""
        current_start_time = 0

        for segment in segments:
            end_time = segment["end"]
            text = segment["text"]

            if end_time - current_start_time >= 60 or segment == segments[-1]:
                # 如果時間間隔超過60秒或是最後一個片段，就輸出當前文本
                start_min, start_sec = divmod(current_start_time, 60)
                end_min, end_sec = divmod(end_time, 60)
                formatted_start_time = f"{int(start_min):02}:{int(start_sec):02}"
                formatted_end_time = f"{int(end_min):02}:{int(end_sec):02}"

                formatted_text = f"({formatted_start_time}~{formatted_end_time}) {current_text.strip()}\n"
                f.write(formatted_text)

                # 重置當前文本和開始時間
                current_text = ""
                current_start_time = end_time

            current_text += " " + text.strip()

    return output_file


def transcribe_folder(input_folder, output_folder):
    # 確認輸出資料夾是否存在，若不存在則創建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 收集所有音訊文件
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 確認是音訊文件
            if file.lower().endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

    # 使用 tqdm 創建進度條並遍歷音訊文件
    with tqdm(total=len(audio_files), desc="Transcribing Files") as pbar:
        for file_path in audio_files:
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")
            
            # 調用轉錄函數
            transcribe_audio(file_path, output_file)
            
            # 更新進度條
            pbar.update(1)

# 確認音訊文件資料夾路徑
input_folder = r"/media/starklab/BACKUP/Podcast_project/mp3/百靈果 News"
output_folder = r"/media/starklab/BACKUP/Podcast_project/轉錄文本存放區/百靈果 News"

# 調用函數進行資料夾中的所有文件轉錄
transcribe_folder(input_folder, output_folder)