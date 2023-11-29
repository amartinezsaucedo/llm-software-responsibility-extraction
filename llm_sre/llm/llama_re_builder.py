from deepeval.metrics import HallucinationMetric, BaseMetric
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from llm_sre.models.configuration.inference import InferenceConfigurationBuilder
from llm_sre.models.configuration.model import ModelConfigurationBuilder
from llm_sre.llm.builder import LLMBuilder
from llm_sre.llm.llmre import LLMRE
from llm_sre.models.configuration.task import TaskConfiguration
from llm_sre.models.example import Example
from llm_sre.utils.prompt import parse_prompt_llama


class LlamaREBuilder(LLMBuilder):
    CONFIGURATION_FILE: str = "./configuration/extraction/configuration.txt"
    _configuration: TaskConfiguration
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _system_message: str
    _output_parser: CommaSeparatedListOutputParser
    _metrics: list[BaseMetric]
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(self):
        self._configuration = TaskConfiguration(self.CONFIGURATION_FILE)

    def set_inference_configuration(self):
        self._inference_configuration = InferenceConfigurationBuilder().build()

    def set_model_configuration(self):
        self._model_configuration = ModelConfigurationBuilder().build()

    def set_system_message(self, system_message: str = None):
        if system_message:
            self._system_message = system_message
        else:
            self._system_message = "You are a concise, precise and truthful assistant that list from a requirement the responsibilities. To identify responsibilities you use the following rules:\nRules:\nA - A responsibility is a verb in base form, 3rd person singular present, or non-3rd person singular present, gerund form, which is associated to a direct object and optionally to the subject performing the action indicated by the verb.\nB - A responsibility is a verb associated to a direct object and optionally to the subject performing the action indicated by the verb, when a verb is actually a phrasal verb modified by preposition, or when using passive voice structures in which the action receives more attention than the subject that performs it; thus, it is rather common that these structures lack the subject of the action.\nC - A responsibility is when a verb is actually a phrasal verb modified by preposition.\n{format_instructions}\nIf none of the rules match, you answer \"None\". List responsibilities concisely and precisely, and do not make up information. Do not add any word and use the requirement text only"
            self._system_message = self._configuration.get_system_message()

    def set_output_parser(self):
        self._output_parser = CommaSeparatedListOutputParser()

    def set_prompt_template(self):
        prompt_template = self._configuration.get_prompt_template()
        examples: list[Example] = self._configuration.get_examples()
        self._prompt_template = HumanMessagePromptTemplate.from_template(parse_prompt_llama(examples, prompt_template))
        self._prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self._system_message),
                self._prompt_template,
            ]
        )

    def set_llm(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self._llm = LlamaCpp(
            callback_manager=callback_manager,
            **self._model_configuration,
            **self._inference_configuration
        )

    def set_metrics(self):
        self._metrics = [HallucinationMetric(minimum_score=0.5)]

    def get_llm(self, evaluate: bool) -> LLMRE:
        return LLMRE(self._llm, self._prompt, self._prompt_template, self._metrics, evaluate)
