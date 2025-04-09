import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 讀取 JSON 文件
with open('/home/starklab/Documents/QA集/qa_dataset.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# 初始化 sentence transformer 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(text1, text2):
    # 計算兩段文本的餘弦相似度
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

results = []

for item in qa_data['questions']:
    question = item['question']
    llm_answer = item['llm_answer']
    ground_truth = item['ground_truth']
    
    # 計算 LLM 回答和 ground truth 的相似度
    similarity = calculate_similarity(llm_answer, ground_truth)
    
    results.append({
        'question': question,
        'similarity_score': similarity
    })

# 計算平均相似度分數
average_similarity = sum(item['similarity_score'] for item in results) / len(results)

# 輸出結果
print(f"Average similarity score: {average_similarity}")
for item in results:
    print(f"Question: {item['question']}")
    print(f"Similarity score: {item['similarity_score']}")
    print()

# 將結果保存到文件，使用自定義的 NumpyEncoder
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        'average_similarity': average_similarity,
        'results': results
    }, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

print("評估完成，結果已保存到 evaluation_results.json")