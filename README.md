# Podcast Retrieval Chatbot with Ollama

以 **LangChain + FAISS + HuggingFace Embeddings + Ollama** 建立的 Podcast 檢索問答系統（RAG），前端使用 **Gradio**。
支援多個向量庫合併檢索、繁體中文輸出、可切換 CPU/GPU、可只看檢索結果以方便除錯/效能測試。

---

## 專案結構

```
.
└─ app/
   ├─ __init__.py
   ├─ main.py                
   ├─ config.py
   ├─ embeddings.py
   ├─ vectorstores.py
   ├─ retriever.py
   ├─ prompts.py
   ├─ relevance.py
   ├─ orchestrator.py
   ├─ service.py
   ├─ llm/
   │  ├─ __init__.py
   │  └─ ollama.py
   └─ ui/
      ├─ __init__.py
      └─ gradio_ui.py
```

* main.py：程式入口，解析參數、組裝各模組並啟動 Gradio UI

* config.py：AppSettings，集中管理設定（向量庫路徑、模型、檢索參數、Ollama 連線等）

* embeddings.py：嵌入模型提供者（BGE-M3），具快取避免重覆初始化

* vectorstores.py：載入或掃描整個資料夾尋找 index.faiss + index.pkl 的 FAISS 向量庫

* retriever.py：MultiStoreRetriever 從多個庫檢索、合併分數、取 top_k

* prompts.py：RAG Prompt 與文件模板

* relevance.py：簡單的關聯度判定（是否延續上一輪話題）

* orchestrator.py：把 Retriever + LLM + Prompt 組裝成 ConversationalRetrievalChain

* service.py：應用層流程（清空記憶、檢索、回答、繁體轉換、時間統計）

* llm/ollama.py：ChatOllamaLLM（繼承 BaseLLM），對接 Ollama HTTP API

* ui/gradio_ui.py：Gradio 介面

---

## 安裝 Ollama

1. 依作業系統安裝 Ollama：[https://ollama.com/download](https://ollama.com/download)
2. 啟動服務：

```bash
ollama serve
```

3. 下載需要的 LLM（預設使用 deepseek-r1:14b）：

```bash
ollama pull deepseek-r1:14b
```

4. 確認是否安裝成功：

```bash
ollama list
```

> 可在 `app/config.py` 中調整 `ollama_model` / `ollama_url`。

---

## 環境需求

* **Python 3.8**（已在 3.8 測試）
* 已安裝 **Ollama** 並啟動 `ollama serve`
* 可選擇使用 GPU（CUDA）或 CPU

---

## 安裝步驟

> 建議使用 conda 建立 `python==3.8` 的虛擬環境。

1. **安裝 PyTorch**
   請依你的 CUDA 版本到官方產生指令：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

2. **安裝 ffmpeg（若需語音轉文字）**

   * Ubuntu/Debian：

     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```
   * Windows：下載 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

3. **安裝 Python 依賴**

   ```bash
   pip install -r requirements.txt
   ```

   （若未含 FAISS）

   * CPU：`pip install faiss-cpu`
   * GPU：`pip install faiss-gpu`

> 請確認 PyTorch / Transformers / FAISS 版本彼此相容（依你的 CUDA 與平台調整）。

---

## 準備向量庫

程式會從 `app/config.py` 的 `AppSettings.parent_vector_dir` 指定的資料夾遞迴掃描；只要子資料夾內**同時**存在：

* `index.faiss`
* `index.pkl`

就會載入為一個向量庫。建議結構：

```
/path/to/vector_store/
├─ show_A/
│  ├─ index.faiss
│  └─ index.pkl
└─ show_B/
   ├─ index.faiss
   └─ index.pkl
```

> 每個 Document 的 `metadata` 若含 `episode_name` / `Podcast_name`，UI 會在回應底部列出「可參考的節目集數」。

---

## 執行

### 一般模式（RAG + LLM 回答）

```bash
python3 main.py
```

### 強制用 CPU 產生嵌入

```bash
python3 main.py --cpu
```

### 只看檢索結果（不呼叫 LLM；除錯/效能測試好用）

```bash
python3 main.py --retrieve-only
```

---

## 設定

在 `app/config.py` 可調整：

* `parent_vector_dir`：向量庫根目錄
* `model_name`：嵌入模型（預設 `BAAI/bge-m3`）
* `top_k` / `fetch_k`：檢索筆數與取前 K
* `ollama_model` / `ollama_url` / `ollama_timeout_sec` / `ollama_stream`

---

## 常見問題（FAQ / Troubleshooting）


* **LangChain Deprecation：`get_relevant_documents`**
  專案已改用 `retriever.invoke(query)`；若你新增程式碼請同樣使用 `invoke()`。

* **Pydantic / 型別驗證錯誤**
  `MultiStoreRetriever` 已把 `top_k` / `fetch_k` 宣告為欄位，請勿在執行期動態新增屬性；且 `Config.arbitrary_types_allowed = True` 已允許 FAISS 型別。

* **Ollama 連線問題**
  檢查 `ollama serve` 是否啟動、`ollama_url` 是否正確；可用 `curl` 測試端點是否可達。

* **啟動或首問很慢**
  嵌入模型已快取；通常是向量庫很多/很大或 `fetch_k` 過大。可先用 `--retrieve-only` 測試、再調整參數。

---

## 擴充

* **換 LLM**：在 `app/llm/` 新增類別（繼承 `BaseLLM` 或包裝為 Runnable），在 `app/main.py` 切換注入
* **換嵌入模型**：改 `AppSettings.model_name` 或自訂 `EmbeddingsProvider`
* **自訂 Prompt**：調整 `app/prompts.py`
* **改 UI**：調整 `app/ui/gradio_ui.py`

