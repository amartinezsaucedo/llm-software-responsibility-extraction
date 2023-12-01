from llm_sre.models.requirement import Requirement
from llm_sre.llm.builders.director import LLMDirector
from llm_sre.llm.builders.llama_rs_builder import LlamaRSBuilder


def sequentialize_responsibilities(requirements_file: str, model_path: str, requirements: list[Requirement], chat: bool,
                                   add_responsibilities_if_missing: bool, evaluate: bool):
    director = LLMDirector(LlamaRSBuilder())
    llm = director.construct_llama_llm_responsibility_sequencing(requirements_file, model_path, chat, evaluate)
    for requirement in requirements:
        llm.prompt(requirement, add_responsibilities_if_missing)
