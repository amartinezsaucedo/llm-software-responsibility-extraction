from typing import Any

from deepeval.evaluator import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma, VectorStore


class LLM:
    _llm: LlamaCpp
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _chain: LLMChain
    _input: dict
    _metrics: list[BaseMetric]
    _vector_store: VectorStore = None
    _evaluate = bool

    def __init__(
        self,
        llm: LlamaCpp,
        prompt: ChatPromptTemplate,
        prompt_template: HumanMessagePromptTemplate,
        metrics: list[BaseMetric],
        evaluate: bool,
        requirements_file_path: str = None,
    ):
        self._llm = llm
        self._prompt = prompt
        self._prompt_template = prompt_template
        self._chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        self._metrics = metrics
        self._evaluate = evaluate
        if requirements_file_path:
            document_loader = UnstructuredFileLoader(requirements_file_path)
            document = document_loader.load()
            self._vector_store = Chroma.from_documents(documents=document, embedding=GPT4AllEmbeddings())

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

    def _post_process_output(self, output: str) -> Any:
        pass

    def _run_test(self, case_input: str, output: str, context: str):
        test_case = LLMTestCase(input=case_input, actual_output=output, context=[context])
        assert_test(test_case, self._metrics)
