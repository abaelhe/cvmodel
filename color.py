import enum
import random
from typing import Union, Tuple

TypeRgb = Tuple[int, int, int]
TypeColor = Union[int, str, TypeRgb]

import numpy as np


COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'silver': (192, 192, 192),
    'gray': (128, 128, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128),
    'navy': (0, 0, 128),
    'brown': (165, 42, 42),
    'orange': (255, 165, 0),
    'gold': (255, 215, 0),
    'khaki': (240, 230, 140),
    'indigo': (75, 0, 130),
    'violet': (238, 130, 238),
    'pink': (255, 192, 203),
    'beige': (245, 245, 220),
    'wheat': (245, 222, 179),
    'chocolate': (210, 105, 30),
    'tan': (210, 180, 140),
    'linen': (250, 240, 230),
}
"""Color aliases to RGB tuples map."""


class _ColorAttr(enum.EnumMeta):
    """
    See:
    https://stackoverflow.com/questions/47353555/how-to-get-random-value-of-attribute-of-enum-on-each-iteration/47353856
    """

    @property
    def RANDOM(self):
        return tuple(random.randint(0, 255) for _ in range(3))


class Color(enum.Enum, metaclass=_ColorAttr):
    """
    Color enum for predefined colors.

    Color.RANDOM returns a random color. Colors can be added together.
    """

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (200, 0, 0)
    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (0, 255, 255)
    dic= {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'silver': (192, 192, 192),
    'gray': (128, 128, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128),
    'navy': (0, 0, 128),
    'brown': (165, 42, 42),
    'orange': (255, 165, 0),
    'gold': (255, 215, 0),
    'khaki': (240, 230, 140),
    'indigo': (75, 0, 130),
    'violet': (238, 130, 238),
    'pink': (255, 192, 203),
    'beige': (245, 245, 220),
    'wheat': (245, 222, 179),
    'chocolate': (210, 105, 30),
    'tan': (210, 180, 140),
    'linen': (250, 240, 230),
  }

    def __add__(self, other):
        if isinstance(other, Color):
            sum = np.asarray(self.value) + np.asarray(other.value)
            normalized = 255 * sum / (sum.max())
            return tuple(map(int, normalized))
        return NotImplemented


CVColor = Union[Color, int, Tuple[int, int, int]]


def _ensure_color_int(color: CVColor) -> Tuple[int, int, int]:
    if isinstance(color, Color):
        color = color.value

    # cv drawing methods need python ints, not numpy ints
    bgr = np.ones(3, dtype=int) * color  # implicit broadcasting
    b, g, r = map(int, bgr)
    return b, g, r

def to_rgb(value: TypeColor) -> Tuple[int, int, int]:
    """Translates the given color value to RGB tuple.

    :param value:

    """
    if isinstance(value, tuple):
        return value

    if isinstance(value, int):
        value = (value, value, value)

    elif isinstance(value, str):
        value = COLORS[value]

    return value
