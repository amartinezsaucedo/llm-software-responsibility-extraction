from llm_sre.llm.llm import LLM


class LLMRE(LLM):
    def prompt(self, requirement: str) -> list:
        output = self._chain({"requirement": requirement})
        return self._get_responsibilities(output["text"])

    def _get_responsibilities(self, output: str) -> list:
        if "AI" in output:
            responsibilities = output.split("AI:")[-1].strip().replace(".", "").split(",")
            responsibilities = [responsibility.strip() for responsibility in responsibilities]
            return list(filter(lambda responsibility: responsibility != "None", responsibilities))
        return []
