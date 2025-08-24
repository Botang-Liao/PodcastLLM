"""
Podcast Q&A (RAG) — SOLID-oriented refactor

This module refactors the original script into cohesive classes that follow SOLID principles.
- Single Responsibility: each class has one well-defined responsibility
- Open/Closed: classes rely on small interfaces; behavior can be extended via subclassing or composition
- Liskov Substitution: abstractions are respected (e.g., BaseRetriever, LLMFacade)
- Interface Segregation: small, focused protocols/ABCs
- Dependency Inversion: high-level orchestration depends on abstractions

Run
----
python main.py             # normal RAG + LLM answer
python main.py --cpu       # force CPU embeddings
python main.py --retrieve-only  # only show retrieved context (no LLM)

Notes
-----
- Keeps HuggingFaceEmbeddings + FAISS.
- Keeps custom ChatOllama LLM, but wrapped behind a small facade.
- Adds explicit, typed settings and graceful error handling.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple

import gradio as gr
import requests
import torch
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLMResult
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from opencc import OpenCC
from pydantic import Field
from langchain.chains import ConversationalRetrievalChain

# ------------------------------
# 1) Configuration & Constants
# ------------------------------

@dataclass(frozen=True)
class AppSettings:
    parent_vector_dir: str = "/home/sylvia2004/projects/PodcastLLM/vector_store"
    model_name: str = "BAAI/bge-m3"
    use_cpu: bool = False
    top_k: int = 5
    fetch_k: int = 100
    ollama_model: str = "deepseek-r1:14b"
    ollama_url: str = "http://163.14.137.59:11434/api/generate"
    ollama_stream: bool = False
    ollama_timeout_sec: int = 120


# --------------------------------
# 2) Embeddings Provider (SRP)
# --------------------------------

class EmbeddingsProvider(Protocol):
    def get(self) -> Any:  # returns an embeddings object compatible with FAISS
        ...

class HFEmbeddingsProvider:
    def __init__(self, model_name: str, use_cpu: bool) -> None:
        self._model_name = model_name
        self._use_cpu = use_cpu
        self._emb = None  # cache

    def get(self):
        if self._emb is None:
            device = "cpu" if self._use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            self._emb = HuggingFaceEmbeddings(model_name=self._model_name, model_kwargs={"device": device})
        return self._emb

# --------------------------------
# 3) Vector Store Loader (SRP)
# --------------------------------

class VectorStoreLoader:
    def __init__(self, embeddings: EmbeddingsProvider) -> None:
        print("Vector Store Loader initialize.")
        self._embeddings = embeddings

    def load_faiss(self, path: str) -> Optional[FAISS]:
        if os.path.exists(path):
            try:
                vs = FAISS.load_local(path, self._embeddings.get(), allow_dangerous_deserialization=True)
                print(f"Loaded vector store from {path}")
                return vs
            except Exception as e:  # noqa: BLE001
                print(f"Error loading vector store from {path}: {e}")
        else:
            print(f"Vector store not found at {path}")
        return None

    def load_all_from_dir(self, parent_dir: str) -> List[FAISS]:
        stores: List[FAISS] = []
        for root, _dirs, files in os.walk(parent_dir):
            if "index.faiss" in files and "index.pkl" in files:
                vs = self.load_faiss(root)
                if vs is not None:
                    stores.append(vs)
        return stores


# --------------------------------
# 4) Retrieval (SRP) + small Strategy
# --------------------------------

class MultiStoreRetriever(BaseRetriever):
    vectorstores: List[FAISS] = Field(default_factory=list)
    top_k: int = 5
    fetch_k: int = 100

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vectorstores: Sequence[FAISS], top_k: int, fetch_k: int, **data):
        super().__init__(vectorstores=list(vectorstores), top_k=top_k, fetch_k=fetch_k, **data)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.retrieve(query)

    def retrieve(self, query: str) -> List[Document]:
        all_results: List[Tuple[Document, float]] = []
        for vs in self.vectorstores:
            _ = vs.max_marginal_relevance_search(query, k=self.fetch_k, fetch_k=self.fetch_k)
            scored = vs.similarity_search_with_score(query, k=self.fetch_k)
            all_results.extend(scored)
        best = heapq.nsmallest(self.top_k, all_results, key=lambda x: x[1])
        return [doc for doc, _ in best]

# --------------------------------
# 5) LLM Facade (SRP)
# --------------------------------

class LLMFacade(Protocol):
    def generate(self, prompts: List[str]) -> LLMResult:
        ...


# imports 需要有
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLMResult

class ChatOllamaLLM(BaseLLM):
    model_name: str
    url: str
    do_stream: bool = False
    timeout_sec: int = 120

    def __init__(self, model_name: str, url: str, stream: bool, timeout_sec: int, **data):
        super().__init__(model_name=model_name, url=url, do_stream=stream, timeout_sec=timeout_sec, **data)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload = {"model": self.model_name, "prompt": prompt, "stream": self.do_stream}
        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout_sec,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                stream=self.do_stream,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[Ollama] connection error: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"[Ollama] HTTP {resp.status_code}: {resp.text[:500]}")

        if not self.do_stream:
            try:
                obj = resp.json()
                return obj.get("response", "")
            except Exception as e:
                raise RuntimeError(f"[Ollama] invalid JSON: {e}. body={resp.text[:500]}")

        full = ""
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    full += obj.get("response", "")
                except json.JSONDecodeError:
                    continue
        finally:
            resp.close()
        return full

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        gens = []
        for p in prompts:
            gens.append([{"text": self._call(p, stop=stop, run_manager=run_manager, **kwargs)}])
        return LLMResult(generations=gens)

    @property
    def _llm_type(self) -> str:
        return "chat_ollama"

# --------------------------------
# 6) Prompt Factory (SRP)
# --------------------------------

class PromptFactory:
    @staticmethod
    def context_prompt() -> ChatPromptTemplate:
        template = (
            """我將作為您的Podcast搜尋引擎。當您向我詢問有關特定Podcast節目或內容的問題時，我將使用RAG（檢索增強生成）技術來回答您的問題。請注意，如果RAG檢索庫中沒有您所需的內容，我將告知您「RAG資料庫內沒有您所需的內容」。我希望您根據這些條件提問。

您的第一句話是「嗨」。

檢索資料信息（包括節目標題）：
{context}

聊天歷史：
{chat_history}

當前問題：
{question}

回答指南：
1. **問題處理**：首先對當前問題進行清晰的 prompt engineering，確保理解問題的核心需求。
2. **信息使用**：僅使用檢索資料中的信息來回答問題。如果資料不足以回答問題，請直接回答「RAG 資料庫沒有您想要的資料」。
3. **回答內容**：
   - **具體內容要點**：回答應包括具體的內容要點。
   - **時間戳**：每個內容要點應附上對應的時間戳。請使用完整的格式，例如（MM:SS~MM:SS）。如果只有一個時間點，則使用（MM:SS）。
   - **節目標題**：最後應提供節目標題（格式：（節目標題：[完整標題]））。
4. **回答格式示例**：
   - 「根據檢索資料，[內容摘要1]（時間戳）。此外，[內容摘要2]（時間戳）。[如有更多內容，繼續列舉]。（節目標題：[完整標題]）」
5. **回答語言和風格**：回答要清楚詳細，使用繁體中文。
6. **資訊限制**：不要添加任何檢索資料中沒有的信息。
7. **格式問題**: 請不要使用刪除線或任何其他特殊格式標記在你的回答中。
8. **記憶**: 如果使用者希望接續前面的問答再次提問，系統應該能夠檢索並提供對話紀錄（chat_history），並根據這些紀錄回答使用者的問題。
請根據上述指南回答問題：
"""
        )
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def document_prompt() -> PromptTemplate:
        return PromptTemplate(
            input_variables=["page_content", "episode_name", "Podcast_name"],
            template="內容: {page_content}\n來源: {episode_name}, {Podcast_name}",
        )


# --------------------------------
# 7) Relevance Policy (SRP)
# --------------------------------

class RelevancePolicy:
    def is_related(self, new_question: str, history: List[Tuple[str, str]]) -> bool:
        if not history:
            return False
        last_question = history[-1][0]
        k1 = set(last_question.lower().split())
        k2 = set(new_question.lower().split())
        overlap = len(k1.intersection(k2))
        threshold = max(1, int(min(len(k1), len(k2)) * 0.3))
        return overlap >= threshold


# --------------------------------
# 8) QA Orchestrator (DIP)
# --------------------------------

class QAOrchestrator:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_facade: ChatOllamaLLM,
        prompt_factory: PromptFactory,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_facade
        self._prompt_factory = prompt_factory
        self._memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self._qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self._llm,  # ChatOllamaLLM implements generate(), compatible
            retriever=self._retriever,
            memory=self._memory,
            combine_docs_chain_kwargs={
                "prompt": self._prompt_factory.context_prompt(),
                "document_variable_name": "context",
                "document_prompt": self._prompt_factory.document_prompt(),
            },
        )

    def clear_memory(self) -> None:
        self._memory.clear()

    def retrieve_only(self, query: str) -> Tuple[str, List[Tuple[str, str]], float]:
        t1 = time.time()
        docs = self._retriever.invoke(query)
        t2 = time.time()
        elapsed = t2 - t1
        uniq: set[Tuple[str, str]] = set()
        ctx_lines: List[str] = []
        for d in docs:
            page = d.page_content.strip()
            ep = d.metadata.get("episode_name", "Unknown Episode")
            pod = d.metadata.get("Podcast_name", "Unknown Podcast")
            ctx_lines.append(f"內容：{page}\n來源：{ep}, {pod}")
            uniq.add((ep, pod))
        src = "\n可參考下方節目集數：\n" + "".join([f"Result {i}: {e}, {p}\n" for i, (e, p) in enumerate(uniq, 1)])
        out = "\n--- 向量資料庫檢索結果 ---\n" + "\n\n".join(ctx_lines[:5]) + "\n\n" + src
        return out, list(uniq), elapsed

    def ask(self, question: str, history: List[Tuple[str, str]]) -> Tuple[str, float]:
        t3 = time.time()
        resp = self._qa_chain.invoke({"question": question, "chat_history": history})
        t4 = time.time()
        return resp.get("answer", ""), (t4 - t3)


# --------------------------------
# 9) Chat Service (application layer)
# --------------------------------

class ChatService:
    def __init__(self, orchestrator: QAOrchestrator, relevance: RelevancePolicy, retrieve_only: bool) -> None:
        self._orch = orchestrator
        self._rel = relevance
        self._retrieve_only = retrieve_only
        self._cc = OpenCC("s2t")

    def handle(self, message: str, history: List[Tuple[str, str]]):
        try:
            if not self._rel.is_related(message, history):
                self._orch.clear_memory()

            ctx, uniq_sources, vec_time = self._orch.retrieve_only(message)
            if self._retrieve_only:
                return self._cc.convert(ctx + f"\n---\n向量檢索時間: {vec_time:.2f} 秒")

            ans, llm_time = self._orch.ask(message, history if history else [])
            src = "\n可參考下方節目集數：\n" + "".join([f"Result {i}: {e}, {p}\n" for i, (e, p) in enumerate(uniq_sources, 1)])
            full = ans + "\n\n" + src
            full = self._cc.convert(full)
            timing = f"\n---\n向量檢索時間: {vec_time:.2f} 秒\nLLM 生成時間: {llm_time:.2f} 秒"
            return full + timing
        except Exception as e:  # noqa: BLE001
            return f"發生錯誤: {e}\n很抱歉，我無法處理您的問題。請再試一次或換個問題。"


# --------------------------------
# 10) UI Layer (SRP)
# --------------------------------

class GradioUI:
    def __init__(self, chat_service: ChatService, program_dir: str) -> None:
        self._svc = chat_service
        self._program_dir = program_dir

    def _get_program_list_text(self) -> str:
        try:
            programs = [n for n in os.listdir(self._program_dir) if os.path.isdir(os.path.join(self._program_dir, n))]
            return "\n".join(f"{i+1}: {p}" for i, p in enumerate(programs))
        except FileNotFoundError:
            return "指定的資料夾不存在。"
        except Exception as e:  # noqa: BLE001
            return f"發生錯誤: {e}"

    def launch(self) -> None:
        def _chat_fn(message: str, history: List[Tuple[str, str]]):
            return self._svc.handle(message, history)

        with gr.Blocks() as iface:
            gr.Markdown(f"## 目前資料庫中的節目有：\n{self._get_program_list_text()}\n\n請在下方提問：")
            gr.ChatInterface(
                _chat_fn,
                title="Podcast Q&A Assistant",
                description="Ask questions about podcast content, and I'll provide answers based on the retrieved information.",
                theme="soft",
                examples=[
                    "還有甚麼節目與這個主題相關",
                    "請告訴我這個節目討論了哪些主題？",
                    "這集節目中有提到哪些重要的觀點？",
                ],
                retry_btn="重試",
                undo_btn="撤銷",
                clear_btn="清除",
            )
        iface.launch(share=True)


# --------------------------------
# 11) Composition Root
# --------------------------------

def build_app(settings: AppSettings, retrieve_only: bool) -> GradioUI:
    emb = HFEmbeddingsProvider(settings.model_name, settings.use_cpu)
    loader = VectorStoreLoader(emb)
    stores = loader.load_all_from_dir(settings.parent_vector_dir)

    retriever = MultiStoreRetriever(stores, top_k=settings.top_k, fetch_k=settings.fetch_k)
    llm = ChatOllamaLLM(
        model_name=settings.ollama_model,
        url=settings.ollama_url,
        stream=settings.ollama_stream,
        timeout_sec=settings.ollama_timeout_sec,
    )
    prompts = PromptFactory()
    orch = QAOrchestrator(retriever, llm, prompts)
    relevance = RelevancePolicy()
    service = ChatService(orch, relevance, retrieve_only=retrieve_only)
    return GradioUI(service, settings.parent_vector_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="Force CPU for embeddings")
    p.add_argument("--retrieve-only", action="store_true", help="Show only retrieved context")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = AppSettings(use_cpu=args.cpu)
    ui = build_app(settings, retrieve_only=args.retrieve_only)
    ui.launch()


if __name__ == "__main__":
    main()
