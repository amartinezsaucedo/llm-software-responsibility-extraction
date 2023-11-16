from llm_sre.models.requirement import Requirement
from llm_sre.llm.director import LLMDirector
from llm_sre.llm.llama_re_builder import LlamaREBuilder


def _extract_requirements(path: str) -> list[Requirement]:
    requirements = []
    file_content = _read_file(path)
    in_requirements_section = False
    for line in file_content:
        if _is_newline(line):
            continue
        if in_requirements_section:
            requirements.append(Requirement(1 + len(requirements), line.strip()))
        in_requirements_section = _is_requirement_section(line)
    return requirements


def _read_file(path: str) -> list:
    with open(path) as file:
        lines = file.readlines()
    return lines


def _is_newline(line: str) -> bool:
    return line.startswith("\n")


def _is_requirement_section(string: str) -> bool:
    return string.startswith("@Requirement")


def _extract_responsibilities_from_requirements(requirements: list[Requirement]) -> list[Requirement]:
    director = LLMDirector(LlamaREBuilder())
    llm = director.construct_llama_llm_responsibility_extraction()
    for requirement in requirements:
        responsibilities = []
        sentences = requirement.get_text().split(".")
        sentences = _preprocess_sentences(sentences)
        requirement.set_sentences(sentences)
        for sentence_id, sentence in enumerate(sentences):
            responsibilities.append((llm.prompt(sentence), sentence_id))
        requirement.add_responsibilities(responsibilities)
    return requirements


def _preprocess_sentences(requirement: list[str]) -> list[str]:
    sentences = []
    for sentence in requirement:
        if sentence:
            sentence = sentence.strip()
            sentence = sentence.lower()
            sentence = sentence.replace("“", "\"")
            sentence = sentence.replace("”", "\"")
            sentence = sentence.split("\"")
            for i, part in enumerate(sentence[:-1]):
                if i % 2 == 1:
                    sentence[i] = part.replace(' ', '_')
            sentences.append("".join(sentence))
    return sentences


def extract_responsibilities_from_file(input_path: str) -> list[Requirement]:
    requirements = _extract_requirements(input_path)
    return _extract_responsibilities_from_requirements(requirements)


if __name__ == "__main__":
    extract_responsibilities_from_file("./cases/example.txt")
