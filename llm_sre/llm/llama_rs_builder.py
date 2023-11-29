from deepeval.metrics import HallucinationMetric, BaseMetric
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from llm_sre.models.configuration.inference import InferenceConfigurationBuilder
from llm_sre.models.configuration.model import ModelConfigurationBuilder
from llm_sre.llm.builder import LLMBuilder
from llm_sre.llm.llmrs import LLMRS
from llm_sre.models.configuration.task import TaskConfiguration
from llm_sre.models.example import Example
from llm_sre.utils.prompt import parse_prompt_llama


class LlamaRSBuilder(LLMBuilder):
    CONFIGURATION_FILE: str = "./configuration/sequencing/configuration.txt"
    _configuration: TaskConfiguration
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _system_message: str
    _metrics: list[BaseMetric]

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
            self._system_message = """You are a concise, precise and truthful assistant that list causal 
            relationships in arrow notation ("cause"->"effect"), in which "cause" and "effect" are responsibilities 
            given by the user and delimited between double quotes. All relationships must have a "cause" and 
            "effect", else do not list them. To identify causal relationships you use the following rules:\n1. The 
            structure of the sentence presents a conditional connotation or a specific expression denoting a sequence 
            of actions.\n2. The sentence has an adverbial time clause and an independent clause, since they slightly 
            present a conditional connotation.\n3. The sentence contains a temporal preposition, which would indicate 
            the presence of an adverbial time clause.\n4. The sentence structure presents a sequence of actions in 
            terms of the combination of a main clause, a subordinating conjunction, and a subordinate clause.\n5. The 
            sentence contains the coordinating conjunctions "and" or "or".\nYou are only allowed to list 
            relationships in which both "cause" and "effect" are in the "Responsibilities" list. A responsibility may 
            be a "cause" and "effect" or none of them. If no causal relationship exists between responsibilities, 
            answer "None". Do not add relationships in which "cause" or "effect" are not in the responsibilities 
            list."""

    def set_output_parser(self):
        self._output_parser = CommaSeparatedListOutputParser()
            self._system_message = self._configuration.get_system_message()

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
        self._metrics = [HallucinationMetric(minimum_score=0.5)]  # , AnswerRelevancyMetric(minimum_score=0.7)]

    def get_llm(self, evaluate: bool) -> LLMRS:
        return LLMRS(self._llm, self._prompt, self._prompt_template, self._metrics, evaluate)
