from llm_sre.llm.builders.builder import LLMBuilder
from llm_sre.llm.llmre import LLMRE
from llm_sre.llm.llmrs import LLMRS
from llm_sre.llm.llmrsyn import LLMRSYN


class LLMDirector:
    _builder: LLMBuilder

    def __init__(self, builder: LLMBuilder):
        self._builder = builder

    def construct_llama_llm_responsibility_extraction(self, requirements_file_path: str, chat: bool,
                                                      evaluate: bool) -> LLMRE:
        self._builder.set_requirements_file(requirements_file_path)
        self._builder.set_model_configuration()
        self._builder.set_inference_configuration()
        self._builder.set_llm()
        self._builder.set_system_message()
        self._builder.set_output_parser()
        if chat:
            self._builder.set_memory()
        self._builder.set_prompt_template()
        self._builder.set_metrics()
        return self._builder.get_llm(evaluate)

    def construct_llama_llm_responsibility_sequencing(self, requirements_file_path: str, chat: bool,
                                                      evaluate: bool) -> LLMRS:
        self._builder.set_requirements_file(requirements_file_path)
        self._builder.set_model_configuration()
        self._builder.set_inference_configuration()
        self._builder.set_llm()
        self._builder.set_system_message()
        if chat:
            self._builder.set_memory()
        self._builder.set_prompt_template()
        self._builder.set_metrics()
        return self._builder.get_llm(evaluate)

    def construct_llama_llm_responsibility_synonym(self, evaluate: bool) -> LLMRSYN:
        self._builder.set_model_configuration()
        self._builder.set_inference_configuration()
        self._builder.set_llm()
        self._builder.set_system_message()
        self._builder.set_prompt_template()
        self._builder.set_metrics()
        return self._builder.get_llm(evaluate)
