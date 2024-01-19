
from typing import Any
from abc import ABC, abstractmethod
import numpy as np

from .option import Option



class Model(ABC):
    """
    Inherit from the class to define your model
    """

    @abstractmethod
    def predict(self, image: np.ndarray)->"ModelOutput":
        raise NotImplementedError()

    def __call__(self, image: np.ndarray) -> Option:
        return Option.wrap(self.predict)(image)



class ModelOutput(ABC):

    """
    Inherit from this class to define your model output\n
    use the to_dict method to generate a json dictionary representation\n
    of you output. (For Apis)

    Use the draw function to define how the output should be drawn onto the screen
    """

    @abstractmethod
    def to_dict(self)->dict:
        raise NotImplementedError()

    @abstractmethod
    def draw(self, image:np.ndarray)->None:
        raise NotImplementedError()


