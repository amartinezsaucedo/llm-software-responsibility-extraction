from llm_sre.llm.llm import LLM


class LLMBuilder:
    def set_inference_configuration(self):
        pass

    def set_model_configuration(self):
        pass

    def set_system_message(self, system_message: str = None):
        pass

    def set_output_parser(self):
        pass

    def set_prompt_template(self):
        pass

    def set_llm(self):
        pass

    def get_llm(self, evaluate: bool) -> LLM:
        pass

    def set_metrics(self):
        pass
