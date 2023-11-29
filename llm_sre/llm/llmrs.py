from random import shuffle
from llm_sre.models.responsibility import Responsibility
from llm_sre.models.requirement import Requirement
from llm_sre.llm.llm import LLM


class LLMRS(LLM):
    _add_responsibilities_if_missing: bool

    def _prepare_prompt(self, requirement: Requirement, add_responsibilities_if_missing: bool):
        self._requirement = requirement
        self._add_responsibilities_if_missing = add_responsibilities_if_missing
        for sentence_index, sentence in enumerate(requirement.get_sentences()):
            responsibilities = requirement.get_responsibilities(sentence_index)
            if len(responsibilities) > 2:
                self._sentence_index = sentence_index
                self._input = {
                    "requirement": f"\"{sentence}\"",
                    "responsibilities": self._format_responsibilities_to_prompt(
                        responsibilities
                    ),
                }

    def _format_responsibilities_to_prompt(self, responsibilities: list[Responsibility]) -> str:
        enumerated_responsibilities = [
            f"\"{responsibility.get_text()}\"" for responsibility in responsibilities
        ]
        shuffle(enumerated_responsibilities)
        return "\n".join(enumerated_responsibilities)

    def _get_causal_relationships_from_output(self, output: str, requirement: Requirement, sentence_index: int,
                                              add_responsibilities_if_missing: bool):
        if "AI" in output:
            causal_relationships = output.split("AI:")[-1].strip().split(",")
    def _post_process_output(self, output: str) -> Requirement:
            for relationship in causal_relationships:
                involved_responsibilities = [relationship.strip().replace("\"", "")
                                             for relationship in relationship.split("->")]
                self._requirement.add_causal_relationship(involved_responsibilities, self._sentence_index,
                                                          self._add_responsibilities_if_missing)
        return self._requirement
