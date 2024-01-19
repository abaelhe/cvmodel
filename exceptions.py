from typing import Union
class InvalidInputError(Exception): ...

class OcvprotoException(Exception):
    """Base ocvproto exception."""


class SourceError(OcvprotoException):
    """Error getting source,"""

def assert_is_type(input, expected_type:Union[type, set[type]]):
    if not isinstance(input, expected_type):
        raise InvalidInputError(f"Expected a '{expected_type}', received '{type(input)}'")

