from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


class LLM:
    _llm: LlamaCpp
    _memory: ConversationBufferMemory
    _prompt: ChatPromptTemplate
    _chain: LLMChain

    def __init__(
        self,
        llm: LlamaCpp,
        prompt: ChatPromptTemplate,
        memory: ConversationBufferMemory,
    ):
        self._llm = llm
        self._memory = memory
        self._prompt = prompt
        self._chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    def prompt(self, **kwargs):
        pass
