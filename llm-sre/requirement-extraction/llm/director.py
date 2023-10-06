from .builder import LLMBuilder
from .llm import LLM


class LLMDirector:
    _builder: LLMBuilder

    def __init__(self, builder: LLMBuilder):
        self._builder = builder

    def construct_llama_llm(self) -> LLM:
        self._builder.set_model_configuration()
        self._builder.set_inference_configuration()
        self._builder.set_system_message()
        self._builder.set_memory()
        self._builder.set_output_parser()
        self._builder.set_prompt_template()
        return self._builder.get_llm()
