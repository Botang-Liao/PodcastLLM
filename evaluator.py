import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 讀取 Excel 檔案
file_path = './dataset/Podcast_QAdataset.xlsx'

df = pd.read_excel(file_path)
 
df.rename(columns={'使用者問題': 'question'}, inplace=True)
df.rename(columns={'LLM回答': 'llm_answer'}, inplace=True)
df.rename(columns={'正確答案': 'correct_answer'}, inplace=True)
df = df[['question', 'llm_answer', 'correct_answer']]
df.dropna(subset=['question', 'llm_answer', 'correct_answer'], inplace=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

# 逐一拿出每一行的LLM回答和正確答案
for index, row in df.iterrows():
    llm_answer = row['llm_answer']
    correct_answer = row['correct_answer']

    # 將LLM回答和正確答案轉換為嵌入向量
    llm_answer_embedding = model.encode(llm_answer)
    correct_answer_embedding = model.encode(correct_answer)

    # 計算餘弦相似度
    similarity = cosine_similarity([llm_answer_embedding], [correct_answer_embedding])[0][0]

    # 將相似度添加到DataFrame中
    df.at[index, 'similarity'] = similarity
    
# 計算平均相似度
average_similarity = df['similarity'].mean()

print(average_similarity)

# 將結果寫入新的Excel檔案
output_file_path = './dataset/Podcast_QAdataset_with_similarity.xlsx'

df.to_excel(output_file_path, index=False)