from typing import Any

from deepeval.evaluator import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore

from llm_sre.models.configuration.task import TaskConfiguration
from llm_sre.utils.prompt import parse_examples


class LLM:
    _llm: LlamaCpp
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _chain: LLMChain
    _input: dict
    _metrics: list[BaseMetric]
    _vector_store: VectorStore = None
    _task_configuration: TaskConfiguration
    _evaluate = bool

    def __init__(
        self,
        llm: LlamaCpp,
        prompt: ChatPromptTemplate,
        prompt_template: HumanMessagePromptTemplate,
        memory: ReadOnlySharedMemory,
        metrics: list[BaseMetric],
        evaluate: bool,
        task_configuration: TaskConfiguration,
        requirements_file_path: str = None,
    ):
        self._llm = llm
        self._prompt = prompt
        self._prompt_template = prompt_template
        self._chain = LLMChain(llm=llm, memory=memory, prompt=prompt, verbose=True)
        self._metrics = metrics
        self._evaluate = evaluate
        self._task_configuration = task_configuration
        if requirements_file_path:
            document_loader = UnstructuredFileLoader(requirements_file_path)
            document = document_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            split_document = text_splitter.split_documents(document)
            self._vector_store = Chroma.from_documents(documents=split_document, embedding=GPT4AllEmbeddings())

    def prompt(self, *args) -> Any:
        self._prepare_prompt(*args)
        prompt = self._prompt_template.format(**self._input).content
        if self._vector_store:
            prompt = self._vector_store.similarity_search(prompt)
            output = self._chain(prompt)
        else:
            output = self._chain(self._input)
        context = parse_examples(self._task_configuration.get_examples())
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
