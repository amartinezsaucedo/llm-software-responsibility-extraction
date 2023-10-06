from .responsibility import Responsibility


class Requirement:
    _id: str
    _text: str
    _responsibilities: list[Responsibility]

    def __init__(self, requirement_id: str, text: str):
        self._id = requirement_id
        self._text = text
        self._responsibilities = []

    def get_text(self) -> str:
        return self._text

    def add_responsibilities(self, responsibilities: list[str]):
        for responsibility in responsibilities:
            self._responsibilities.append(Responsibility(responsibility.strip()))