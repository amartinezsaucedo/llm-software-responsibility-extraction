from deepeval.metrics import BaseMetric
from langchain import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    MessagesPlaceholder

from llm_sre.llm.llm import LLM
from llm_sre.models.configuration.task import TaskConfiguration
from llm_sre.models.example import Example
from llm_sre.utils.prompt import parse_prompt_llama


class LLMBuilder:
    _configuration: TaskConfiguration
    _llm: LlamaCpp
    _requirements_file: str
    _inference_configuration: dict
    _model_configuration: dict
    _memory: ReadOnlySharedMemory = None
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _system_message: str
    _metrics: list[BaseMetric]

    def set_requirements_file(self, requirements_file: str):
        self._requirements_file = requirements_file

    def set_inference_configuration(self):
        pass

    def set_model_configuration(self, model_path: str):
        pass

    def set_system_message(self, system_message: str = None):
        if system_message:
            self._system_message = system_message
        else:
            self._system_message = self._configuration.get_system_message()

    def set_output_parser(self):
        pass

    def set_memory(self):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for example in self._configuration.get_examples():
            memory.save_context({"input": example.get_input()}, {"output": example.get_output()})
        self._memory = ReadOnlySharedMemory(memory=memory)

    def set_prompt_template(self):
        prompt_template = self._configuration.get_prompt_template()
        examples: list[Example] = self._configuration.get_examples()
        if self._memory:
            self._prompt_template = HumanMessagePromptTemplate.from_template(prompt_template)
        else:
            self._prompt_template = HumanMessagePromptTemplate.from_template(
                parse_prompt_llama(examples, prompt_template))
        messages = [
            SystemMessagePromptTemplate.from_template(self._system_message),
            self._prompt_template,
        ]
        if self._memory:
            messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
        self._prompt = ChatPromptTemplate(messages=messages)

    def set_llm(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self._llm = LlamaCpp(
            callback_manager=callback_manager,
            **self._model_configuration,
            **self._inference_configuration
        )

    def get_llm(self, evaluate: bool) -> LLM:
        pass

    def set_metrics(self):
        pass
