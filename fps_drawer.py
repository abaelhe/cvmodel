import time
from typing import Callable
import numpy as np
import cv2


def _default_draw_fps(fps):
    return f"{int(fps)} FPS"

class FpsDrawer:
    """
    This class is used to draw the frames per secondn(fps) on the screen
    """

    def __init__(self, draw_fps_fn:Callable[[float], str]=_default_draw_fps, org=(7, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color= (100, 255, 0), thickness=3, lineType=cv2.LINE_AA) -> None:

        self.__counter = -1
        self._start = time.perf_counter()
        self._tick()
        self.org = org
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.color = color
        self.thickness = thickness
        self.lineType = lineType
        self.draw_fps_fn = draw_fps_fn

    def _tick(self):
        self.__counter += 1
        self._end = time.perf_counter()

    def get_fps(self):
        return self.__counter / (self._end - self._start)

    def draw(self, image:np):
        self._tick()
        cv2.putText(image, self.draw_fps_fn(self.get_fps()), self.org, self.fontFace, self.fontScale, self.color, self.thickness, self.lineType)


