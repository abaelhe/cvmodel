from typing import Any, Union


class NoneAccessError(Exception): ...

class Option:

    """
    This class can use used to wrap operations that can lead to an Exception to\n
    or an unwated None result to provide some level of safety.\n

    """

    def __init__(self, data:Union[None, Any], exception:Union[Exception, None]=None) -> None:
        self.__data = data
        self.exception = exception

    def is_ok(self)->bool:
        """
        Checks if the result of the last operation has succeeded\n
        It must always be used before calling 'unwrap'

        """

        return not self.__data is None

    def unwrap(self):
        """
        Use this method the retrieve the last result. Must call is_ok before using it.
        """
        if not self.is_ok():
            raise NoneAccessError("Tried to access the value while the last operation failed. You must call the is_ok method before calling unwrap.")
        return self.__data

    def wrap(func):

        """
        Use the as a wrapper for your function\n


        from: def do_something(...): ...\n
              x = do_something(...)

        to: @Option.wrap
            def do_somthing(...): ...
            result = do_something(...)
            if result.is_ok():
                x = result.unwrap()

        """

        def wrapper(*args, **kwargs):

            data = None
            exception = None

            try:
                data = func(*args, **kwargs)

            except Exception as e:
                exception = e

            if isinstance(data, Option):
                return data

            return Option(data=data, exception=exception)

        return wrapper
