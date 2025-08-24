from __future__ import annotations
from langchain.prompts import ChatPromptTemplate, PromptTemplate

class PromptFactory:
    @staticmethod
    def context_prompt() -> ChatPromptTemplate:
    template = 