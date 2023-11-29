from typing import Any

from deepeval.evaluator import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain.llms import LlamaCpp
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


class LLM:
    _llm: LlamaCpp
    _memory: BaseChatMemory
    _prompt: ChatPromptTemplate
    _chain: LLMChain
    _input: dict
    _metrics: list[BaseMetric]
    _evaluate = bool

    def __init__(
        self,
        llm: LlamaCpp,
        prompt: ChatPromptTemplate,
        memory: BaseChatMemory,
        prompt_template: HumanMessagePromptTemplate,
        metrics: list[BaseMetric],
        evaluate: bool,
    ):
        self._llm = llm
        self._memory = memory
        self._prompt = prompt
        self._chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
        self._prompt_template = prompt_template
        self._metrics = metrics
        self._evaluate = evaluate
            self._vector_store = Chroma.from_documents(documents=document, embedding=GPT4AllEmbeddings())

    def prompt(self, **args):
    def prompt(self, *args) -> Any:
        self._prepare_prompt(*args)
        prompt = self._prompt_template.format(**self._input).content
        if self._vector_store:
            prompt = self._vector_store.similarity_search(prompt)
            output = self._chain(prompt)
        else:
            output = self._chain(self._input)
        context = '' # get_buffer_string(self._memory.memory.chat_memory.messages) TODO
        if self._evaluate:
            self._run_test(prompt, output["text"], context)
        return self._post_process_output(output["text"])

    def _prepare_prompt(self, *args):
        pass
