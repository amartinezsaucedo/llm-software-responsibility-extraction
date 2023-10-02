from typing import List
from models.requirement import Requirement
from models.responsibility import Responsibility
from llm.director import LLMDirector
from llm.llama_builder import LlamaBuilder


def extract_requirements(path: str) -> List[Requirement]:
    requirements = list()
    file_content = read_file(path)
    in_requirements_section = False
    for line in file_content:
        if is_requirement_section(line):
            in_requirements_section = True
            continue
        if in_requirements_section:
            requirements.append(Requirement(f"REQ-{1 + len(requirements)}", line.strip()))
    return requirements


def read_file(path: str) -> list:
    with open(path) as file:
        lines = file.readlines()
    return lines


def is_requirement_section(string: str) -> bool:
    return string.startswith("@Requirement")


def extract_responsibilities_from_requirements(requirements: List[Requirement]) -> List[Responsibility]:
    director = LLMDirector(LlamaBuilder())
    llm = director.construct_llama_llm()
    responsibilities = []
    for requirement in requirements:
        llm.prompt(requirement.get_text())
    return responsibilities


if __name__ == "__main__":
    req = extract_requirements("./cases/example.txt")
    res = extract_responsibilities_from_requirements(req)
