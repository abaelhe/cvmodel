import os
from typing import Optional, Union, Iterator, Callable, Protocol, Any
from threading import Thread
from queue import Queue
from contextlib import contextmanager

import cv2 as cv
import numpy as np

from .image import _error_if_image_empty
from .fps_drawer import FpsDrawer
from .model_schema import Model, ModelOutput
from .exception import InvalidInputError, assert_is_type


class VideoStream(Protocol):
    """
    Interface to generate a stream of images

    def read(self)->tuple[bool, Optional[np.ndarray]]:
        TODO: Add your logic

    def release(self):
        TODO: Add your logic
    """
    def read(self)->tuple[bool, Optional[np.ndarray]]:
        pass

    def release(self):
        pass


class ImageFolderStream:
    def __init__(self, dir_path:str, exts:Union[None, str, set[str]] = None, repeat:bool=True, abs_path:bool=True) -> None:

        if not os.path.isdir(dir_path):
            raise Exception(f"{dir_path} is not a directory")
        self.__exts =exts and (set([exts]) if isinstance(exts, str) else exts) or None
        self.__repeat = repeat
        self.__index = 0
        self.__paths = []

        for path in os.listdir(dir_path):
            if self.exts is not None and path.rpartition('.')[-1] not in self.exts:
                continue
            full_path = os.path.join(dir_path, path)
            if bool(abs_path):
                full_path = os.path.abspath(full_path)
            self.__paths.append(full_path)

        self.__max_index = len(self.__paths) -1


    def read(self)->tuple[bool, np.ndarray]:
        if self.__index == self.__max_index:
            if not self.__repeat:
                return False, None
            else:
                self.__index = 0
        image = cv.imread(self.__paths[self.__index])
        self.__index += 1
        return True, image

    def release(self):
        self.__index = self.__max_index
        self.__repeat = False

class VideoStreamer:
    """
    Use this class the read videos from path or from webcam
    Use the on_next_frame method as a decorator to wrap

    """

    def __init__(self, cap: VideoStream, window_name="video", waitkey=1) -> None:

      self.__cap = cap
      self.__tasks = []
      self.__pre_tasks = []
      self.window_name = window_name
      self.waitkey = waitkey


    def from_webcam(cam_index:int=0,window_name:str="video", waitkey:int=1):

        assert_is_type(cam_index, int)

        return VideoStreamer(
           cap = cv.VideoCapture(cam_index),
           window_name=window_name,
           waitkey=waitkey
       )

    def from_folder(path: str, exts=["jpg", "jpeg", "png"], repeat:bool=True, window_name:str="video", waitkey:int=1):

        assert_is_type(path, str)

        return VideoStreamer(
           cap = ImageFolderStream(path, exts, repeat=repeat),
           window_name=window_name,
           waitkey=waitkey
       )

    def on_next_frame(self, shape:Optional[tuple[int,int]]=None):
        """
        Use as a decorator to wrap your function\n
        The cam window is drawn automatically after the function has been called.\n
        Example: \n

        stream =  VideoStreamer.from_webcam()
        @stream.on_next_frame()
        def func(image):
             ...


        """
        if not shape is None:
           self.__pre_tasks.append(lambda img: cv.resize(img, shape, interpolation=cv.INTER_AREA))

        def inner(func:Callable[[np.ndarray], None]):
            self.__tasks.append(func)

        return inner


    def start_with_model(self, model: Model, fps_drawer:Optional[FpsDrawer]=FpsDrawer()):

        if fps_drawer:
            @self.on_next_frame()
            def fn(image: np.ndarray):
                output = model.predict(image)
                output.draw(image)
                fps_drawer.draw(image)
        else:
            @self.on_next_frame()
            def fn(image: np.ndarray):
                output = model.predict(image)
                output.draw(image)

        self.start()

    def from_video_input(path, window_name="video", waitkey=1):

        assert_is_type(path, str)

        return VideoStreamer(
            cap=cv.VideoCapture(path),
           window_name=window_name,
           waitkey=waitkey
        )

    def __step(self, frame):
        for fn in self.__tasks:

            fn(frame)

            cv.imshow(self.window_name, frame)
            cv.waitKey(self.waitkey)

    def start(self):
        """
        Using the method to start loading the next frames
        """

        queue = Queue()

        def update():

            while True:
                ret, frame = self.__cap.read()
                if ret:
                   queue.put(frame)
                else:
                   queue.put(None)

        thread = Thread(target=update)
        thread.daemon = True
        thread.start()

        while True:

            if  queue.empty(): continue

            frame = queue.get()
            if frame is None: return self.close()

            for task in self.__pre_tasks:
                frame = task(frame)

            self.__step(frame)

    def close(self):
        self.__cap.release()
        cv.destroyAllWindows()



"""
Convenience functions for reading and writing videos.

Usage:

>>> import cv2 as cv
>>> with load_video("path/to/file") as video:
>>>    for frame in video:
>>>        cv.imshow("Frame", frame)
>>>        cv.waitKey(1)
"""

import cv2 as cv
import numpy as np



class VideoCapture(cv.VideoCapture):
    """A video capture object for displaying videos.

    For normal use, use :func:`.load_camera` and :func:`.load_camera` instead.
    """

    def __init__(self, source: Union[int, str]):
        """
        The object can be created using either an index of a connected camera, or a
        filename of a video file.

        :param source: Either index of camera or filename of video file.
        """
        super().__init__(source)

    def __iter__(self):
        ok, current = self.read()
        if not ok:
            raise ValueError(f"Could not read video.")

        next_ok, next = self.read()

        yield current

        # If next frame is also good
        if next_ok:
            current = next

            while True:
                next_ok, next = self.read()

                yield current

                current = next

                if not next_ok:
                    return


@contextmanager
def load_camera(index: int = 0) -> Iterator[Any]:
    """Open a camera for video capturing.

    :param index: Index of the camera to open.

                  For more details see `cv2.VideoCapture(index) documentation`_

    .. _cv2.VideoCapture(index) documentation : https://docs.opencv.org/3.4.5/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
    """
    video = VideoCapture(index)
    if not video.isOpened():
        raise ValueError(f"Could not open camera with index {index}")
    try:
        yield video
    finally:
        video.release()


@contextmanager
def load_video(filename: str) -> Iterator[Any]:
    """
    Open a video file

    :param filename: It an be:

                     * Name of video file
                     * An image sequence
                     * A URL of a video stream

                     For more details see `cv2.VideoCapture(filename) documentation`_

    .. _cv2.VideoCapture(filename) documentation: https://docs.opencv.org/3.4.3/d8/dfe/classcv_1_1VideoCapture.html#a85b55cf6a4a50451367ba96b65218ba1
    """

    video = VideoCapture(filename)
    if not video.isOpened():
        raise ValueError(f"Could not open video with filename {filename}")
    try:
        yield video
    finally:
        video.release()


def read_frames(
    video: VideoCapture, start: int = 0, stop: Optional[int] = None, step: int = 1
) -> Iterator[np.ndarray]:
    """Read frames of a video object.

    Start, stop and step work as built-in range.

    :param video: Video object to read from.
    :param start: Frame number to skip to.
    :param stop: Frame number to stop reading, exclusive.
    :param step: Step to iterate over frames. Similar to range's step. Must be greater than 0.
    """
    if stop is not None and start >= stop:
        raise ValueError(f"from_frame ({start}) must be less than to_frame ({stop})")
    if step <= 0 or not isinstance(step, int):
        raise ValueError(f"Step must be an integer greater than 0: {step}")

    ok, current = video.read()
    if not ok:
        raise ValueError(f"Could not read video.")

    next_ok, next = video.read()

    # Skip frames until from_frame
    counter = 0
    while start > counter:
        if not next_ok:
            raise ValueError(
                f"Not enough frames to skip to frame {start}. File ended at frame {counter}."
            )
        current = next

        next_ok, next = video.read()
        counter += 1

    yield current

    # If next frame is also good
    if next_ok:
        current = next

        while True:
            for i in range(step):
                # +1 to make to_frame exclusive
                if counter + 1 == stop:
                    return

                next_ok, next = video.read()
                counter += 1

                if not next_ok:
                    if i == step - 1:
                        yield current
                    return

            yield current

            current = next

            if not next_ok:
                return


class VideoWriter(object):
    """
    A video writer for writing videos, using OpenCV's `cv.VideoWriter`.

    The video writer is lazy, in that it waits to receive the first frame, before determining
    the frame size for the video writer. This is in contrast to OpenCV's video writer, which
    expects a frame size up front.
    """

    def __init__(
        self, filename: str, fps: int = None, capture: Any = None, fourcc: str = "MJPG"
    ):
        """
        Either `fps` or `capture` must be provided.

        For additional documentation, see `cv2.VideoWriter documentation`_

        .. _cv2.VideoWriter documentation: https://docs.opencv.org/3.4.5/dd/d9e/classcv_1_1VideoWriter.html

        :param filename: Name of the output video file.
        :param fps: Framerate of the created video stream.
        :param capture: A capture object from cv2.VideoCapture or :func:`load_video`. Used to retrieve
                        fps if `fps` is not provided.
        :param fourcc: 4-character code of codec used to compress the frames. See
                       `documentation <https://docs.opencv.org/3.4.5/dd/d9e/classcv_1_1VideoWriter.html#ac3478f6257454209fa99249cc03a5c59>`_
        """
        self.filename = filename
        self.fourcc = fourcc

        if fps is not None:
            self.fps = fps
        elif capture is not None:
            self.fps = capture.get(cv.CAP_PROP_FPS)
        else:
            raise ValueError("Either `fps` or `capture` must be provided")

        self.writer = None
        self.frame_shape = None

    def write(self, frame):
        """Write a frame to the video.

        The frame must be the same size each time the frame is written.

        :param frame: Image to be written
        """
        _error_if_image_empty(frame)
        if self.writer is None:
            self.frame_shape = frame.shape
            self.writer = cv.VideoWriter()
            self.writer.open(
                self.filename,
                cv.VideoWriter_fourcc(*self.fourcc),
                self.fps,
                (frame.shape[1], frame.shape[0]),
                frame.ndim == 3,
            )
        else:
            if frame.shape != self.frame_shape:
                raise ValueError(
                    f"frame.shape {frame.shape} does not match previous shape {self.frame_shape}"
                )
        # Write to video
        self.writer.write(frame)
