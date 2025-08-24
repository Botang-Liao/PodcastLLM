import argparse
from config import AppSettings
from embeddings import HFEmbeddingsProvider
from vectorstores import VectorStoreLoader
from retriever import MultiStoreRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory
from orchestrator import QAOrchestrator
from relevance import RelevancePolicy
from service import ChatService
from ui.gradio_ui import GradioUI



# imports 需要有
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLMResult


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
