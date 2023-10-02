class Requirement:
    _id: str
    _text: str

    def __init__(self, requirement_id: str, text: str):
        self._id = requirement_id
        self._text = text

    def get_text(self) -> str:
        return self._text
