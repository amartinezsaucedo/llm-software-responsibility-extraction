from deepeval.metrics import HallucinationMetric
from langchain.llms import LlamaCpp

from llm_sre.llm.builders.builder import LLMBuilder
from llm_sre.llm.llmrs import LLMRS
from llm_sre.models.configuration.inference import InferenceConfigurationBuilder
from llm_sre.models.configuration.model import ModelConfigurationBuilder
from llm_sre.models.configuration.task import TaskConfiguration


class LlamaRSBuilder(LLMBuilder):
    CONFIGURATION_FILE: str = "./configuration/sequencing/configuration.txt"
    _configuration: TaskConfiguration
    _llm: LlamaCpp

    def __init__(self):
        self._configuration = TaskConfiguration(self.CONFIGURATION_FILE)

    def set_inference_configuration(self):
        self._inference_configuration = InferenceConfigurationBuilder().build()

    def set_model_configuration(self):
        self._model_configuration = ModelConfigurationBuilder().build()

    def set_metrics(self):
        self._metrics = [HallucinationMetric(minimum_score=0.5)]  # , AnswerRelevancyMetric(minimum_score=0.7)]

    def get_llm(self, evaluate: bool) -> LLMRS:
        return LLMRS(self._llm, self._prompt, self._prompt_template, self._memory, self._metrics, evaluate,
                     self._configuration, self._requirements_file)
