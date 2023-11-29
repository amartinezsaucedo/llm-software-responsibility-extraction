from deepeval.metrics import HallucinationMetric, BaseMetric
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
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
    _output_parser: CommaSeparatedListOutputParser
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    _metrics: list[BaseMetric]

    def set_inference_configuration(self):
        self._inference_configuration = {
            "echo": True,
            "frequency_penalty": 0.0,
            "logprobs": None,
            "max_tokens": 4096,
            "mirostat_mode": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "n_parts": -1,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.1,
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

    def set_prompt_template(self):
        self._prompt_template = HumanMessagePromptTemplate.from_template(
            "Sentence:\n{requirement}\nResponsibilities:\n{responsibilities}"
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
                    requirement='"If a student accesses the system then the student can add a new course or drop an '
                                'added course"',
                    responsibilities='"access system"\n"add course"\n"drop course"',
                ).content
            },
            {"output": '"access system"->"add course", "access system"->"drop course"'},
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(
                    requirement='"After the student adds a course, the system sends the transaction information to '
                                'the billing system"',
                    responsibilities='"send information"\n"add course"',
                ).content
            },
            {"output": '"add course"->"send information"'},
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(
                    requirement='"The system shall list the classes that a student can attend"',
                    responsibilities='"list classes"',
                ).content
            },
            {"output": "None"},
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(
                    requirement='"The student can click on the "list courses" button for listing its added courses"',
                    responsibilities='"list courses"\n"click button"',
                ).content
            },
            {"output": '"click button"->"list courses"'},
        )
        memory.save_context(
            {
                "input": self._prompt_template.format(
                    requirement='"When the student drops a course, a confirmation dialog is displayed on the screen. '
                                'Then, if the student confirms deletion, course is deleted and marked as inactive"',
                    responsibilities='"drop course"\n"display dialog"\n"confirm deletion"\n"delete course"\n"mark '
                                     'course"',
                ).content
            },
            {
                "output": '"drop course"->"display dialog", "confirm deletion"->"delete course", "confirm '
                          'deletion"->"mark course"'
            },
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
        self._metrics = [HallucinationMetric(minimum_score=0.5)]  # , AnswerRelevancyMetric(minimum_score=0.7)]

    def get_llm(self, evaluate: bool) -> LLMRS:
        return LLMRS(self._llm, self._prompt, self._prompt_template, self._metrics, evaluate)
