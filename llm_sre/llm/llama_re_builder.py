from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from llm_sre.llm.builder import LLMBuilder
from llm_sre.llm.llmre import LLMRE


class LlamaREBuilder(LLMBuilder):
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _memory: ConversationBufferMemory
    _prompt: ChatPromptTemplate
    _system_message: str
    _output_parser: CommaSeparatedListOutputParser
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
            self._system_message = "You are a concise and precise assistant that list from the following requirement the responsibilities. To identify responsibilities use the following rules:\nRules:\nA - A responsibility is a verb in base form, 3rd person singular present, or non-3rd person singular present, gerund form, which is associated to a direct object and optionally to the subject performing the action indicated by the verb.\nB - A responsibility is a verb associated to a direct object and optionally to the subject performing the action indicated by the verb, when a verb is actually a phrasal verb modified by preposition, or when using passive voice structures in which the action receives more attention than the subject that performs it; thus, it is rather common that these structures lack the subject of the action.\nC - A responsibility is when a verb is actually a phrasal verb modified by preposition.\n{format_instructions}\nAnswer concisely and precisely, do not add information or context"

    def set_output_parser(self):
        self._output_parser = CommaSeparatedListOutputParser()

    def set_prompt_template(self):
        format_instructions = self._output_parser.get_format_instructions()
        self._prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self._system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{requirement}"),
            ],
            partial_variables={"format_instructions": format_instructions},
        )

    def set_memory(self):
        self._memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self._memory.save_context(
            {
                "input": "If a student accesses the system then the student can add a new course or drop an added course"
            },
            {"output": "access system, add course, drop course"},
        )
        self._memory.save_context(
            {
                "input": "After the student adds a course, the system sends the transaction information to the billing system"
            },
            {"output": "add course, send information to billing system"},
        )

    def get_llm(self) -> LLMRE:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self._llm = LlamaCpp(
            callback_manager=callback_manager,
            **self._model_configuration,
            **self._inference_configuration
        )
        return LLMRE(self._llm, self._prompt, self._memory)
