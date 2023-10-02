from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence


class LLM:
    _chain: RunnableSequence

    def __init__(self, llm: LlamaCpp, prompt: ChatPromptTemplate):
        self._chain = prompt | llm

    def prompt(self, requirement: str):
        return self._chain.invoke({"input": requirement})


