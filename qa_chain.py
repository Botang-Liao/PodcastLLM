from langchain.chains import ConversationalRetrievalChain
from llm.chat_ollama import ChatOllama
from memory.conversation_buffer import make_memory
from prompts.templates import make_system_prompt, make_document_prompt
from retrievers.multi_store_mmr import MultiStoreMMRRetriever
from embeddings.hf_embeddings import create_embeddings
from vectorstores.faiss_loader import load_vectorstores_from_root
from config.settings import settings

def build_chain():
    llm = ChatOllama()
    memory = make_memory()

    embeddings = create_embeddings()
    vectorstores = load_vectorstores_from_root(settings.VECTORSTORE_ROOT, embeddings)
    retriever = MultiStoreMMRRetriever(
        vectorstores=vectorstores,
        top_k=settings.TOP_K,
        fetch_k=settings.FETCH_K,
    )

    prompt = make_system_prompt()
    document_prompt = make_document_prompt()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
            "document_prompt": document_prompt,
        },
    )
    return chain, retriever
