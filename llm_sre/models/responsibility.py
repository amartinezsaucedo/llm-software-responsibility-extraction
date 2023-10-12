class Responsibility:
    _id: int
    _text: str
    
    def __init__(self, responsibility_id: int, text: str):
        self._id = responsibility_id
        self._text = text

    def get_id(self) -> str:
        return str(self._id)
    
    def get_text(self) -> str:
        return self._text 
    
    def is_responsibility(self, responsibility: str) -> bool:
        return self._text == responsibility
    
    def __repr__(self):
        return self._text
