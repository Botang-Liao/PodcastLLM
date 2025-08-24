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
