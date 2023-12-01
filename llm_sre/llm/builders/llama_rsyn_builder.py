from deepeval.metrics import HallucinationMetric, BaseMetric
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from llm_sre.models.configuration.inference import InferenceConfigurationBuilder
from llm_sre.models.configuration.model import ModelConfigurationBuilder
from llm_sre.llm.builders.builder import LLMBuilder
from llm_sre.llm.llmrsyn import LLMRSYN


class LlamaRSYNBuilder(LLMBuilder):
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _memory: ReadOnlySharedMemory = None
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _system_message: str
    _metrics: list[BaseMetric]
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def set_inference_configuration(self):
        self._inference_configuration = InferenceConfigurationBuilder().build()

    def set_model_configuration(self):
        self._model_configuration = ModelConfigurationBuilder().build()

    def set_system_message(self, system_message: str = None):
        if system_message:
            self._system_message = system_message
        else:
            self._system_message = ("You are a truthful and concise assistant that analyzes sentences to answer "
                                    "whether they are referring to the same concept or not in given contexts. You "
                                    "always reply 'True' or 'False'")

    def set_prompt_template(self):
        self._prompt_template = HumanMessagePromptTemplate.from_template("""Do these two responsibilities mean the 
        same according to these two contexts?:\n{ responsibilities}\n{contexts}""")
        #  Reply 'True' or 'False' only
        self._prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self._system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                self._prompt_template,
            ],
            input_variables=["chat_history", "responsibilities", "contexts"],
        )

    def set_memory(self):
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(responsibilities="Responsibility 1: 'list "
                                                                       "classes'\nResponsibility 2: 'list "
                                                                       "courses'\n", contexts="Context 1: 'The system "
                                                                                              "shall list the classes "
                                                                                              "that a student can "
                                                                                              "attend'\nContext 2: "
                                                                                              "'The student can click "
                                                                                              "on the \"list "
                                                                                              "courses\" button for "
                                                                                              "listing its added "
                                                                                              "courses'")
            },
            {"output": "True"},
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(responsibilities="Responsibility 1: 'add course'\Responsibility "
                                                                       "2: 'send information to billing system'\n",
                                                      contexts="Context 1: 'After the student adds a course, "
                                                               "the system sends the transaction information to the "
                                                               "billing system'\nContext 2: 'After the student adds a "
                                                               "course, the system sends the transaction information "
                                                               "to the billing system'")
            },
            {"output": "False"},
        )
        self._memory = ReadOnlySharedMemory(memory=memory)

    def set_llm(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self._llm = LlamaCpp(
            callback_manager=callback_manager,
            **self._model_configuration,
            **self._inference_configuration
        )

    def set_metrics(self):
        self._metrics = [HallucinationMetric(minimum_score=0.5)]

    def get_llm(self, evaluate: bool) -> LLMRSYN:
        return LLMRSYN(self._llm, self._prompt, self._prompt_template, self._memory, self._metrics, evaluate, None)
