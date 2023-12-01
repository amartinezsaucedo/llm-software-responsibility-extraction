class Example:
    _input: str
    _output: str

    def __init__(self, model_input: str, model_output: str):
        self._input = model_input
        self._output = model_output

    def get_input(self) -> str:
        return self._input

    def get_output(self) -> str:
        return self._output

    def get_text(self) -> str:
        return f"- Human: {self._input}\n- AI: {self._output}"


