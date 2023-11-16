class Responsibility:
    _id: int
    _text: str
    _sentence_id: int
    
    def __init__(self, responsibility_id: int, text: str, sentence_id: int):
        self._id = responsibility_id
        self._text = text
        self._sentence_id = sentence_id

    def get_id(self) -> str:
        return str(self._id)
    
    def get_text(self) -> str:
        return self._text
    
    def get_sentence_id(self) -> int:
        return self._sentence_id

    def is_responsibility(self, responsibility: str) -> bool:
        return self._text == responsibility
    
    def __repr__(self):
        return self._text
