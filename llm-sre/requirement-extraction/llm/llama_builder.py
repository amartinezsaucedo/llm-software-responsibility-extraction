from .builder import LLMBuilder
from .llm import LLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import (FewShotChatMessagePromptTemplate, ChatPromptTemplate)


class LlamaBuilder(LLMBuilder):
    _llm: LlamaCpp
    _inference_configuration: dict
    _model_configuration: dict
    _system_message: str
    _prompt: ChatPromptTemplate

    def set_inference_configuration(self):
        self._inference_configuration = {
            "echo": True,
            "frequency_penalty": 0.0,
            "logprobs": None,
            "max_tokens": 0,
            "mirostat_mode": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "n_parts": -1,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.1,
            "stop": [],
            "stopping_criteria": None,
            "stream": True,
            "top_k": 40,
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
            "n_gpu_layers": 1,
            "n_threads": None,
            "numa": False,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "use_mmap": False,
            "use_mlock": True,
            "verbose": True,
            "vocab_only": False
        }

    def set_system_message(self, system_message: str = None):
        if system_message:
            self._system_message = system_message
        else:
            self._system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully " \
                                   "as possible, while being safe.  Your answers should not include any harmful, " \
                                   "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure " \
                                   "that your responses are socially unbiased and positive in nature.\n\nIf a " \
                                   "question does not make any sense, or is not factually coherent, explain why " \
                                   "instead of answering something not correct. If you don't know the answer to a " \
                                   "question, please don't share false information. "

    def set_prompt_template(self):
        instruction = """Identify from the following requirement the responsibilities. To identify responsibilities 
        use the following rules: \nRules: \nA - A responsibility is a verb in base form, 3rd person singular present, 
        or non-3rd person singular present, gerund form, which is associated to a direct object and optionally to the 
        subject performing the action indicated by the verb. \nB - A responsibility is a verb associated to a direct 
        object and optionally to the subject performing the action indicated by the verb, when a verb is actually a 
        phrasal verb modified by preposition, or when using passive voice structures in which the action receives 
        more attention than the subject that performs it; thus, it is rather common that these structures lack the 
        subject of the action. \nC - A responsibility is when a verb is actually a phrasal verb modified by 
        preposition". \nRequirement: "{0}" """

        examples = [
            {"input": instruction.format("If a student accesses the system then the student can add a new course or "
                                         "drop an added course."), "output": "access system, add course, drop course"},
            {"input": instruction.format("After the student adds a course, the system sends the transaction "
                                         "information to the billing system."), "output": "add course, "
                                                                                          "send information to "
                                                                                          "billing system"}
        ]

        example_prompt = ChatPromptTemplate.from_messages([("user", "{input}"), ("assistant", "{output}")])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_message),
            few_shot_prompt,
            ("user", "{input}")
        ])

    def get_llm(self) -> LLM:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self._llm = LlamaCpp(callback_manager=callback_manager, **self._model_configuration,
                             **self._inference_configuration)
        return LLM(self._llm, self._prompt)
