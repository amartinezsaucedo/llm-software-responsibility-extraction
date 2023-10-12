from random import shuffle
from typing import List
from llm_sre.models.responsibility import Responsibility
from llm_sre.models.requirement import Requirement
from llm_sre.llm.llm import LLM


class LLMRS(LLM):
    def prompt(self, requirement: Requirement) -> Requirement:
        responsibilities = requirement.get_responsibilities()
        if len(responsibilities) > 1:
            output = self._chain(
                {
                    "requirement": requirement.get_text(),
                    "responsibilities": self._format_responsibilities_to_prompt(
                        responsibilities
                    ),
                }
            )
            self._get_causal_relationships_from_output(output["text"], requirement)
        return requirement

    def _format_responsibilities_to_prompt(
        self, responsibilities: List[Responsibility]
    ) -> str:
        enumerated_responsibilities = [
            responsibility.get_text() for responsibility in responsibilities
        ]
        shuffle(enumerated_responsibilities)
        return "\n".join(enumerated_responsibilities)

    def _get_causal_relationships_from_output(
        self, output: str, requirement: Requirement
    ):
        if "AI" in output:
            causal_relationships = output.split("AI:")[-1].strip().split(",")
            for relationship in causal_relationships:
                involved_responsibilities = relationship.split("->")
                requirement.add_causal_relationship(involved_responsibilities)
