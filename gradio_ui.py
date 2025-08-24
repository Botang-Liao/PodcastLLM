import time
import os
from chains.qa_chain import build_chain
from utils.relevance import is_related
from utils.text_norm import to_traditional
from config.settings import settings

def _list_programs(folder_path: str) -> str:
    try:
        items = [n for n in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, n))]
        return "\n".join(f"{i+1}: {n}" for i, n in enumerate(items))
    except FileNotFoundError:
        return "指定的資料夾不存在。"
    except Exception as e:
        return f"發生錯誤: {e}"

def make_chat_function(chain, retriever):
    def chat_fn(message, history):
        try:
            related = is_related(message, history)
            if not related:
                chain.memory.clear()

            t1 = time.time()
            results = retriever.invoke(message)
            t2 = time.time()
            vector_t = t2 - t1

            unique_sources = set()
            context_snippets = []
            for doc in results:
                page_content = (doc.page_content or "").strip()
                episode_name = doc.metadata.get("episode_name", "Unknown Episode")
                podcast_name = doc.metadata.get("Podcast_name", "Unknown Podcast")
                context_snippets.append(f"內容：{page_content}\n來源：{episode_name}, {podcast_name}")
                unique_sources.add((episode_name, podcast_name))

            sources_str = "\n可參考下方節目集數：\n"
            for idx, (ep, pod) in enumerate(unique_sources, 1):
                sources_str += f"Result {idx}: {ep}, {pod}\n"

            if settings.RETRIEVE_ONLY:
                out = "\n--- 向量資料庫檢索結果 ---\n" + "\n\n".join(context_snippets[:5]) + "\n\n" + sources_str
                out += f"\n---\n向量檢索時間: {vector_t:.2f} 秒"
                return to_traditional(out)

            t3 = time.time()
            resp = chain.invoke({"question": message, "chat_history": history if related else []})
            t4 = time.time()
            llm_t = t4 - t3

            answer = resp.get("answer", "")
            full = f"{answer}\n\n{sources_str}"
            full = to_traditional(full)
            full += f"\n---\n向量檢索時間: {vector_t:.2f} 秒\nLLM 生成時間: {llm_t:.2f} 秒"
            return full

        except Exception as e:
            return f"發生錯誤: {e}\n很抱歉，我無法處理您的問題。請再試一次或換個問題。"

    return chat_fn

def build_gradio_blocks():
    import gradio as gr
    chain, retriever = build_chain()
    chat_fn = make_chat_function(chain, retriever)

    with gr.Blocks() as app:
        gr.Markdown(f"## 目前資料庫中的節目有：\n{_list_programs('./vector_store')}\n\n請在下方提問：")
        gr.ChatInterface(
            chat_fn,
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
    return app
