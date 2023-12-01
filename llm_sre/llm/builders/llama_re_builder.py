from deepeval.metrics import HallucinationMetric

from llm_sre.models.configuration.inference import InferenceConfigurationBuilder
from llm_sre.models.configuration.model import ModelConfigurationBuilder
from llm_sre.llm.builders.builder import LLMBuilder
from llm_sre.llm.llmre import LLMRE
from llm_sre.models.configuration.task import TaskConfiguration


class LlamaREBuilder(LLMBuilder):
    CONFIGURATION_FILE: str = "./configuration/extraction/configuration.txt"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(self):
        self._configuration = TaskConfiguration(self.CONFIGURATION_FILE)

    def set_inference_configuration(self):
        self._inference_configuration = InferenceConfigurationBuilder().build()

    def set_model_configuration(self, model_path: str):
        self._model_configuration = ModelConfigurationBuilder().set_model_path(model_path).build()

    def set_metrics(self):
        self._metrics = [HallucinationMetric(minimum_score=0.5)]

    def get_llm(self, evaluate: bool) -> LLMRE:
        return LLMRE(self._llm, self._prompt, self._prompt_template, self._memory, self._metrics, evaluate,
                     self._configuration, self._requirements_file)
