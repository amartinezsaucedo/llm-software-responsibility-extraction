class Responsibility:
    _verb: str
    _direct_object: str
    _long_direct_object: str
    _text: str
    
    def __init__(self, text: str, verb: str = None,  direct_object: str = None, long_direct_object: str = None):
        self._text = text
        self._verb = verb
        self._direct_object = direct_object
        self._long_direct_object = long_direct_object
