from typing import List
from llm_sre.models.requirement import Requirement
from llm_sre.llm.director import LLMDirector
from llm_sre.llm.llama_rs_builder import LlamaRSBuilder


def sequentialize_responsibilities(requirements: List[Requirement]):
    director = LLMDirector(LlamaRSBuilder())
    llm = director.construct_llama_llm_responsibility_sequencing()
    for requirement in requirements:
        llm.prompt(requirement)
