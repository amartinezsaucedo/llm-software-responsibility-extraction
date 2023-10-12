from llm_sre.llm.llm import LLM


class LLMRE(LLM):
    def prompt(self, requirement: str) -> list:
        output = self._chain({"requirement": requirement})
        return self._get_responsibilities(output["text"])

    def _get_responsibilities(self, output: str) -> list:
        if "AI" in output:
            return output.split("AI:")[-1].strip().split(",")
        return []
