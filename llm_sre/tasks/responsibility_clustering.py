from llm_sre.models.requirement import Requirement
from llm_sre.llm.director import LLMDirector
from llm_sre.llm.llama_rsyn_builder import LlamaRSYNBuilder
from llm_sre.models.responsibility import Responsibility


def cluster_responsibilities(requirements: list[Requirement], evaluate: bool):
    director = LLMDirector(LlamaRSYNBuilder())
    llm = director.construct_llama_llm_responsibility_synonym(evaluate)
    responsibilities = [
        (responsibility, responsibility.get_sentence_id(),
         requirement.get_sentence_by_index(responsibility.get_sentence_id()))
        for requirement in requirements
        for responsibility in requirement.get_responsibilities()
    ]
    for (responsibility_a, sentence_a_id, context_a) in responsibilities:
        for (responsibility_b, sentence_b_id, context_b) in responsibilities:
            if sentence_a_id != sentence_b_id and llm.prompt([responsibility_a, responsibility_b],
                                                             [context_a, context_b]):
                _replace_requirement(responsibility_a, responsibility_b, requirements)


def _replace_requirement(original_responsibility: Responsibility, copied_responsibility: Responsibility,
                         requirements: list[Requirement]):
    requirements_with_copied_responsibility = [
        requirement
        for requirement in requirements if requirement.find_responsibility(copied_responsibility.get_text())
    ]
    for requirements_to_modify in requirements_with_copied_responsibility:
        requirements_to_modify.remove_responsibility(copied_responsibility)
        requirements_to_modify.add_responsibility(original_responsibility)
