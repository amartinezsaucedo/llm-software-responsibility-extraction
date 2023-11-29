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
from llm_sre.llm.builder import LLMBuilder
from llm_sre.llm.llmrsyn import LLMRSYN


class LlamaRSYNBuilder(LLMBuilder):
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _memory: ReadOnlySharedMemory
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _system_message: str
    _metrics: list[BaseMetric]
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def set_inference_configuration(self):
        self._inference_configuration = {
            "echo": True,
            "frequency_penalty": 0.0,
            "logprobs": None,
            "max_tokens": 512,
            "mirostat_mode": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "n_parts": -1,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.2,
            "stop": [],
            "stopping_criteria": None,
            "stream": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.0,
            "tfs_z": 1.0,
        }

    def set_model_configuration(self):
        self._model_configuration = {
            "embedding": False,
            "f16_kv": True,
            "last_n_tokens_size": 64,
            "logits_all": False,
            "lora_base": None,
            "lora_path": None,
            "low_vram": False,
            "main_gpu": 0,
            "model_path": "./model/gguf-model.bin",
            "mul_mat_q": True,
            "n_batch": 512,
            "n_ctx": 4096,
            "n_gpu_layers": 2,
            "n_threads": None,
            "numa": False,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "use_mmap": False,
            "use_mlock": True,
            "verbose": True,
            "vocab_only": False,
        }

    def set_system_message(self, system_message: str = None):
        if system_message:
            self._system_message = system_message
        else:
            self._system_message = ("You are a truthful and concise assistant that analyzes sentences to answer "
                                    "whether they are referring to the same concept or not in given contexts. You "
                                    "always reply 'True' or 'False'")

    def set_prompt_template(self):
        self._prompt_template = ("Do these two responsibilities mean the same according to these two contexts?:\n{"
                                 "responsibilities}\n{contexts}")
        #  Reply 'True' or 'False' only
        self._prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self._system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(self._prompt_template),
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
        return LLMRSYN(self._llm, self._prompt, self._prompt_template, self._memory, self._metrics, evaluate)
