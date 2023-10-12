from llm_sre.llm.builder import LLMBuilder
from llm_sre.llm.llmre import LLMRE
from llm_sre.llm.llmrs import LLMRS


class LLMDirector:
    _builder: LLMBuilder

    def __init__(self, builder: LLMBuilder):
        self._builder = builder

    def construct_llama_llm_responsibility_extraction(self) -> LLMRE:
        self._builder.set_model_configuration()
        self._builder.set_inference_configuration()
        self._builder.set_system_message()
        self._builder.set_output_parser()
        self._builder.set_prompt_template()
        self._builder.set_memory()
        return self._builder.get_llm()

    def construct_llama_llm_responsibility_sequencing(self) -> LLMRS:
        self._builder.set_model_configuration()
        self._builder.set_inference_configuration()
        self._builder.set_system_message()
        self._builder.set_output_parser()
        self._builder.set_prompt_template()
        self._builder.set_memory()
        return self._builder.get_llm()
