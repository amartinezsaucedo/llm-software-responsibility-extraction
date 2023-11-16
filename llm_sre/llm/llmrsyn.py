from llm_sre.models.responsibility import Responsibility
from llm_sre.llm.llm import LLM


class LLMRSYN(LLM):
    def prompt(self, responsibilities: list[Responsibility], contexts: list[str]) -> bool:
        if len(responsibilities) == 2:
            output = self._chain(
                {
                    "responsibilities": self._format_responsibilities_to_prompt(
                        responsibilities
                    ),
                    "contexts": self._format_context_to_prompt(
                        contexts
                    ),
                }
            )
            return self._get_response_from_output(output["text"])
        return False

    def _format_responsibilities_to_prompt(self, responsibilities: list[Responsibility]) -> str:
        return "\n".join([f"Sentence {responsibility_id}: '{responsibility.get_text()}'"
                          for (responsibility_id, responsibility) in enumerate(responsibilities)])

    def _format_context_to_prompt(self, contexts: list[str]) -> str:
        return "\n".join([f"Context {context_id}: '{context}'"
                          for (context_id, context) in enumerate(contexts)])

    def _get_response_from_output(self, output: str) -> bool:
        if "AI" in output:
            return output.split("AI:")[-1].strip().lower() == "True"
        return False
