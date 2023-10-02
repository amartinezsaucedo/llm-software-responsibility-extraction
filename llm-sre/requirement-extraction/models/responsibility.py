class Responsibility:
    _verb: str
    _direct_object: str
    _long_direct_object: str

    def __init__(self, verb: str,  direct_object: str, long_direct_object: str):
        self._verb = verb
        self._direct_object = direct_object
        self._long_direct_object = long_direct_object
