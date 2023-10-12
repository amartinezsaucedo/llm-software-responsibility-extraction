from typing import List
from llm_sre.models.requirement import Requirement
from llm_sre.llm.director import LLMDirector
from llm_sre.llm.llama_re_builder import LlamaREBuilder


def _extract_requirements(path: str) -> List[Requirement]:
    requirements = []
    file_content = _read_file(path)
    in_requirements_section = False
    for line in file_content:
        if _is_requirement_section(line):
            in_requirements_section = True
            continue
        if in_requirements_section:
            requirements.append(Requirement(1 + len(requirements), line.strip()))
    return requirements


def _read_file(path: str) -> list:
    with open(path) as file:
        lines = file.readlines()
    return lines


def _is_requirement_section(string: str) -> bool:
    return string.startswith("@Requirement")


def _extract_responsibilities_from_requirements(requirements: List[Requirement]) -> List[Requirement]:
    director = LLMDirector(LlamaREBuilder())
    llm = director.construct_llama_llm_responsibility_extraction()
    for requirement in requirements:
        requirement.add_responsibilities(llm.prompt(requirement.get_text()))
    return requirements


def extract_responsibilities_from_file(path: str) -> List[Requirement]:
    requirements = _extract_requirements(path)
    return _extract_responsibilities_from_requirements(requirements)

if __name__ == "__main__":
    extract_responsibilities_from_file("./cases/example.txt")
