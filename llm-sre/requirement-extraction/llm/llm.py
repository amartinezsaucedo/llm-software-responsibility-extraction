from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser


class LLM:
    _llm: LlamaCpp
    _memory: ConversationBufferMemory
    _chain: LLMChain
    _output_parser: CommaSeparatedListOutputParser

    def __init__(self, llm: LlamaCpp, prompt: ChatPromptTemplate, memory: ConversationBufferMemory, output_parser: CommaSeparatedListOutputParser):
        self._llm = llm
        self._memory = memory
        self._output_parser = output_parser
        self._chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    def prompt(self, requirement: str) -> list: 
        output = self._chain({"requirement": requirement})
        return self._get_responsibilities(output["text"])

    def _get_responsibilities(self, output: str) -> list:
        if "AI" in output:
            return output.split("AI:")[-1].strip().split(",")
        return []
