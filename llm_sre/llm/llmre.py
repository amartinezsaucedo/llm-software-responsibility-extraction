from llm_sre.llm.llm import LLM


class LLMRE(LLM):

    def _prepare_prompt(self, requirement: str):
        self._input = {"requirement": requirement}

    def _post_process_output(self, output: str) -> list:
        prefix = "AI:"
        if not prefix in output:
            prefix = "ANSWER:"
        if prefix in output:
            output = output.split("QUESTION:")[0]  # Sometimes, LLM repeats question, although first part of
            # the response is correct
            responsibilities = output.split(prefix)[-1].strip().replace(".", "").replace("\"", "").split(",")
            responsibilities = [responsibility.strip() for responsibility in responsibilities]
            return list(filter(lambda responsibility: responsibility and
                                                      responsibility != "None", responsibilities))
        return []
