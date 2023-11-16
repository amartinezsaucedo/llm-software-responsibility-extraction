from llm_sre.models.requirement import Requirement
from llm_sre.llm.director import LLMDirector
from llm_sre.llm.llama_rs_builder import LlamaRSBuilder


def sequentialize_responsibilities(requirements: list[Requirement], add_responsibilities_if_missing: bool):
    director = LLMDirector(LlamaRSBuilder())
    llm = director.construct_llama_llm_responsibility_sequencing()
    for requirement in requirements:
        llm.prompt(requirement, add_responsibilities_if_missing)
