import os
from opencc import OpenCC

# 初始化簡體轉繁體的轉換器
cc = OpenCC('s2t')  # s2t 代表簡體到繁體

# 指定要處理的資料夾路徑
folder_path = r"/media/starklab/BACKUP/Podcast_project/轉錄文本存放區/史塔克實驗室" # 替換成你的資料夾路徑


# 瀏覽資料夾內的所有 .txt 檔案

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        # 讀取文件內容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 將內容從簡體翻譯成繁體
        converted_content = cc.convert(content)
        
        # 將翻譯後的內容覆寫回原文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(converted_content)


print("done")
