from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from llm_sre.llm.builder import LLMBuilder
from llm_sre.llm.llmrs import LLMRS


class LlamaRSBuilder(LLMBuilder):
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _memory: ReadOnlySharedMemory
    _prompt: ChatPromptTemplate
    _prompt_template: HumanMessagePromptTemplate
    _system_message: str
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
            "n_gpu_layers": 3,
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
            self._system_message = 'You are a concise and precise assistant that identify causal relationships from a given sentence in arrow notation using the following format "cause"->"effect". To represent each "cause" and "effect" use the responsibilities provided by the user.'

    def set_prompt_template(self):
        self._prompt_template = HumanMessagePromptTemplate.from_template(
            'Identify causal relationships of this sentence "{requirement}" in arrow notation using the following responsibilities to identify "cause"->"effect": {responsibilities}'
        )
        self._prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self._system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                self._prompt_template,
            ],
            input_variables=["chat_history", "requirement", "responsibilities"],
        )

    def set_memory(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(
                    requirement="If a student accesses the system then the student can add a new course or drop an added course",
                    responsibilities="access system\nadd course\ndrop course",
                ).content
            },
            {"output": "access system->add course, access system->drop course"},
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(
                    requirement="After the student adds a course, the system sends the transaction information to the billing system",
                    responsibilities="send information to billing system\nadd course",
                ).content
            },
            {"output": "add course->send information to billing system"},
        )
        self._memory = ReadOnlySharedMemory(memory=memory)

    def get_llm(self) -> LLMRS:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self._llm = LlamaCpp(
            callback_manager=callback_manager,
            **self._model_configuration,
            **self._inference_configuration
        )
        return LLMRS(self._llm, self._prompt, self._memory)
