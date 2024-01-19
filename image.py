import builtins
import enum
import cmath
from dataclasses import dataclass
from typing import Tuple, Union, Iterator, Optional, Sequence, Dict, Any

import cv2 as cv
import numpy as np

PairOfFloats = Tuple[float, float]


@dataclass
class Point:
    """Model class for a point.

    Points can be negated, added, subtracted and iterated over, yielding x, y.
    """

    x: float
    y: float

    def __add__(self, other: Any) -> "Point":
        if self.__class__ is other.__class__:
            return Point(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__len__") and len(other) == 2:
            return Point(self.x + other[0], self.y + other[1])
        return NotImplemented

    def __radd__(self, other: Any) -> "Point":
        return self + other

    def __sub__(self, other: Any) -> "Point":
        if self.__class__ is other.__class__:
            return Point(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__len__") and len(other) == 2:
            return Point(self.x - other[0], self.y - other[1])
        return NotImplemented

    def __rsub__(self, other: Any) -> "Point":
        return -self + other

    def __neg__(self) -> "Point":
        return Point(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y))

    def cartesian(self) -> PairOfFloats:
        """
        :return: The point represented as cartesian coordinates
        """
        return self.x, self.y

    def polar(self) -> PairOfFloats:
        """
        :return: The point represented as polar coordinates
        """
        return cmath.polar(complex(self.x, self.y))

    @classmethod
    def origin(cls) -> "Point":
        """
        :return: Return the origin point, Point(0, 0)
        """
        return cls(0, 0)

    @property
    def norm(self) -> float:
        """
        Return the absolute L2 norm of the point. Alias for `cvw.norm(point)`.

        :return: The absolute L2 norm of the point
        """
        from .misc_functions import norm as norm_func

        return norm_func(self)


CVPoint = Union[Point, Tuple[int, int]]


@dataclass
class Rect:
    """Model class of a rectangle.

    Rectangles can be iterated over, yielding x, y, width, height.

    Rectangles can also be divided. The division is applied to x, y, the width
    and the height of the rectangle. The makes the rectangle fit to an
    image shrinked by the same factor.

    A test whether or not a point is located inside the rectangle can be checked
    by the `in` keyword: `if point in rect`.
    """

    x: float
    y: float
    width: float
    height: float

    def __init__(
        self, x: float, y: float, width: float, height: float, *, padding: float = 0
    ):
        """If padding is given, the rectangle will be `padding` pixels larger in each
        of its sides. The padding can be negative.

        :param x: The top-left x coordinate.
        :param y: The top-left y coordinate.
        :param width: The width of the rectangle.
        :param height: The height of the rectangle.
        :param padding: The padding to be applied to the rectangle.
        """
        self.x = x - padding
        self.y = y - padding
        self.width = width + padding * 2
        self.height = height + padding * 2
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Rect must have width and height >= 0: {self}")

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.width, self.height))

    def __truediv__(self, other: float) -> "Rect":
        if isinstance(other, (int, float)):
            return Rect(
                self.x / other, self.y / other, self.width / other, self.height / other
            )
        return NotImplemented

    def __floordiv__(self, other: float) -> "Rect":
        if isinstance(other, int):
            return Rect(
                self.x // other,
                self.y // other,
                self.width // other,
                self.height // other,
            )
        return NotImplemented

    def __contains__(self, point: CVPoint) -> bool:
        if isinstance(point, tuple):
            point = Point(*point)
        if isinstance(point, Point):
            return (
                self.x <= point.x < self.x + self.width
                and self.y <= point.y < self.y + self.height
            )
        raise ValueError("Must be called with a point or a 2-tuple (x, y)")

    def __or__(self, other: "Rect") -> "Rect":
        if self.__class__ is not other.__class__:
            return NotImplemented

        # Same as OpenCV's implementation
        if self.empty():
            return other
        elif other.empty():
            return self

        x = min(self.x, other.x)
        y = min(self.y, other.y)
        width = max(self.x + self.width, other.x + other.width) - x
        height = max(self.y + self.height, other.y + other.height) - y

        return Rect(x, y, width, height)

    def __and__(self, other: "Rect") -> Optional["Rect"]:
        if self.__class__ is not other.__class__:
            return NotImplemented

        # Same as OpenCV's implementation
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        width = min(self.x + self.width, other.x + other.width) - x
        height = min(self.y + self.height, other.y + other.height) - y

        if width <= 0 or height <= 0:
            return None

        return Rect(x, y, width, height)

    @property
    def tl(self) -> Point:
        """
        :return: The top-left corner of the rectangle.
        """
        return Point(self.x, self.y)

    @property
    def tr(self) -> Point:
        """
        :return: The top-right corner of the rectangle.
        """
        return Point(self.x + self.width, self.y)

    @property
    def bl(self) -> Point:
        """
        :return: The bottom-left corner of the rectangle.
        """
        return Point(self.x, self.y + self.height)

    @property
    def br(self) -> Point:
        """
        :return: The bottom-right corner of the rectangle.
        """
        return Point(self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Point:
        """
        :return: The center point of the rectangle.
        """
        return Point(self.x + (self.width / 2), self.y + (self.height / 2))

    @property
    def area(self) -> float:
        """
        :return: The area of the rectangle.
        """
        return self.width * self.height

    @property
    def slice(self) -> Tuple[builtins.slice, builtins.slice]:
        """Creates a slice of the rectangle, to be used on a 2-D numpy array.

        For example `image[rect.slice] = 255` will fill the area represented by
        the rectangle as white, in a gray-scale, uint8 image.

        :return: The slice of the rectangle.
        """
        return (
            slice(int(self.y), int(self.y) + int(self.height)),
            slice(int(self.x), int(self.x) + int(self.width)),
        )

    def cartesian_corners(self) -> Tuple[PairOfFloats, PairOfFloats]:
        """Yields the rectangle as top-left and bottom-right points, as used in cv2.rectangle.

        :return: The top-left and bottom-right corners of the rectangle as cartesian two-tuples.
        """
        return self.tl.cartesian(), self.br.cartesian()

    def empty(self) -> bool:
        """
        :return: Whether or not the rectangle is empty.
        """
        return self.width <= 0 or self.height <= 0


CVRect = Union[Rect, Tuple[int, int, int, int]]


@dataclass
class Contour:
    """Model class for a contour.

    The points come from cv2.findContours(). Using
    :func:`.find_external_contours` is preferred.
    """

    points: np.ndarray

    def __post_init__(self) -> None:
        self._moments: Optional[Dict[str, float]] = None
        self._bounding_rect: Optional[Rect] = None

    @property
    def area(self) -> float:
        """Return the area computed from cv.moments(points).

        :return: The area of the contour
        """
        if self._moments is None:
            self._moments = cv.moments(self.points)
        return self._moments["m00"]

    @property
    def bounding_rect(self) -> Rect:
        """Return the bounding rectangle around the contour.

        Uses cv2.boundingRect(points).

        :return: The bounding rectangle of the contour
        """
        if self._bounding_rect is None:
            self._bounding_rect = Rect(*cv.boundingRect(self.points))
        return self._bounding_rect

    @property
    def center(self) -> Point:
        """Return the center point of the area.

        Due to skewed densities, the center
        of the bounding rectangle is preferred to the center from moments.

        :return: The center of the bounding rectangle
        """
        return self.bounding_rect.center

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key: Any) -> Union[np.ndarray, float]:
        if isinstance(key, int):
            return self.points[key, 0]
        if len(key) > 2:
            raise ValueError(f"Too many indices: {len(key)}")
        return self.points[key[0], 0, key[1]]

    def __setitem__(self, key: Any, value: float) -> None:
        if isinstance(key, int):
            self.points[key, 0] = value
        if len(key) > 2:
            raise ValueError(f"Too many indices: {len(key)}")
        self.points[key[0], 0, key[1]] = value



class MorphShape(enum.Enum):
    """Enum for determining shape in morphological operations.

    Alias for OpenCV's morph enums.
    """

    RECT: int = cv.MORPH_RECT
    CROSS: int = cv.MORPH_CROSS
    CIRCLE: int = cv.MORPH_ELLIPSE


class AngleUnit(enum.Enum):
    """
    Enum for which angle unit to use.
    """

    RADIANS = enum.auto()
    DEGREES = enum.auto()


def line_iterator(image: np.ndarray, p1: Point, p2: Point) -> np.ndarray:
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points.

    Credit: https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator

    :param image: The image being processed
    :param p1: The first point
    :param p2: The second point
    :return: An array that consists of the coordinates and intensities of each pixel on the line.
             (shape: [numPixels, 3(5)], row = [x,y, intensity(b, g, r)]), for gray-scale(bgr) image.
    """
    # define local variables for readability
    imageH = image.shape[0]
    imageW = image.shape[1]
    # P1X = P1[0]
    # P1.y = P1[1]
    # P2X = P2[0]
    # P2.y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = p2.x - p1.x
    dY = p2.y - p1.y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    color_chls = 1 if image.ndim == 2 else 3
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 2 + color_chls), dtype=np.int32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = p1.y > p2.y
    negX = p1.x > p2.x
    if p1.x == p2.x:  # vertical line segment
        itbuffer[:, 0] = p1.x
        if negY:
            itbuffer[:, 1] = np.arange(p1.y - 1, p1.y - dYa - 1, -1, dtype=np.int32)
        else:
            itbuffer[:, 1] = np.arange(p1.y + 1, p1.y + dYa + 1, dtype=np.int32)
    elif p1.y == p2.y:  # horizontal line segment
        itbuffer[:, 1] = p1.y
        if negX:
            itbuffer[:, 0] = np.arange(p1.x - 1, p1.x - dXa - 1, -1, dtype=np.int32)
        else:
            itbuffer[:, 0] = np.arange(p1.x + 1, p1.x + dXa + 1, dtype=np.int32)
    else:  # diagonal line segment
        # TODO: error here when drawing from bottom right to top left diagonal.
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX / dY
            if negY:
                itbuffer[:, 1] = np.arange(p1.y - 1, p1.y - dYa - 1, -1, dtype=np.int32)
            else:
                itbuffer[:, 1] = np.arange(p1.y + 1, p1.y + dYa + 1, dtype=np.int32)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - p1.y)).astype(np.int) + p1.x
        else:
            slope = dY / dX
            if negX:
                itbuffer[:, 0] = np.arange(p1.x - 1, p1.x - dXa - 1, -1, dtype=np.int32)
            else:
                itbuffer[:, 0] = np.arange(p1.x + 1, p1.x + dXa + 1, dtype=np.int32)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - p1.x)).astype(np.int) + p1.y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    # Get three values if color image
    num_channels = 2 if image.ndim == 2 else slice(2, None, None)
    itbuffer[:, num_channels] = image[itbuffer[:, 1], itbuffer[:, 0]]

    return itbuffer


def rect_intersection(rect1: Rect, rect2: Rect) -> Optional[Rect]:
    """
    Calculate the intersection between two rectangles.

    :param rect1: First rectangle
    :param rect2: Second rectangle
    :return: A rectangle representing the intersection between `rect1` and `rect2`
             if it exists, else None.
    """
    top = min(rect1, rect2, key=lambda x: x.tl.y)
    bottom = max(rect2, rect1, key=lambda x: x.tl.y)
    if top.br.y < bottom.tl.y or top.br.x < bottom.bl.x or top.bl.x > bottom.br.x:
        return None

    tl = Point(max(top.tl.x, bottom.tl.x), bottom.tl.y)

    width = min(bottom.br.x, top.br.x) - tl.x
    height = min(bottom.br.y, top.br.y) - tl.y

    return Rect(tl.x, tl.y, width, height)


def find_external_contours(image: np.ndarray) -> Tuple[Contour, ...]:
    """Find the external contours in the `image`.

    Alias for `cv2.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)`

    :param image: The image in with to find the contours
    :return: A tuple of Contour objects
    """
    _error_if_image_empty(image)
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    contours = contours if contours is not None else ()
    return (*map(Contour, contours),)


def dilate(
    image: np.ndarray,
    kernel_size: int,
    shape: MorphShape = MorphShape.RECT,
    iterations: int = 1,
) -> np.ndarray:
    """Dilate `image` with `kernel_size` and `shape`.

    :param image: Image to be dilated
    :param kernel_size: Kernel size to dilate with
    :param shape: Shape of kernel
    :param iterations: Number of iterations to perform dilation
    :return: The dilated image
    """
    _error_if_image_empty(image)
    return cv.dilate(
        image,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def erode(
    image: np.ndarray,
    kernel_size: int,
    shape: MorphShape = MorphShape.RECT,
    iterations: int = 1,
) -> np.ndarray:
    """Erode `image` with `kernel_size` and `shape`.

    :param image: Image to be eroded
    :param kernel_size: Kernel size to erode with
    :param shape: Shape of kernel
    :param iterations: Number of iterations to perform erosion
    :return: The eroded image
    """
    _error_if_image_empty(image)
    return cv.erode(
        image,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def morph_open(
    image: np.ndarray,
    kernel_size: int,
    shape: MorphShape = MorphShape.RECT,
    iterations=1,
) -> np.ndarray:
    """Morphologically open `image` with `kernel_size` and `shape`.

    :param image: Image to be opened
    :param kernel_size: Kernel size to open with
    :param shape: Shape of kernel
    :param iterations: Number of iterations to perform opening
    :return: The opened image
    """
    _error_if_image_empty(image)
    return cv.morphologyEx(
        image,
        cv.MORPH_OPEN,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def morph_close(
    image: np.ndarray,
    kernel_size: int,
    shape: MorphShape = MorphShape.RECT,
    iterations=1,
) -> np.ndarray:
    """Morphologically close `image` with `kernel_size` and `shape`.

    :param image: Image to be closed
    :param kernel_size: Kernel size to close with
    :param shape: Shape of kernel
    :param iterations: Number of iterations to perform closing
    :return: The closed image
    """
    _error_if_image_empty(image)
    return cv.morphologyEx(
        image,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def normalize(
    image: np.ndarray, min: int = 0, max: int = 255, dtype: np.dtype = None
) -> np.ndarray:
    """Normalize image to range [`min`, `max`].

    :param image: Image to be normalized
    :param min: New minimum value of image
    :param max: New maximum value of image
    :param dtype: Output type of image. Default is same as `image`.
    :return: The normalized image
    """
    _error_if_image_empty(image)
    normalized = np.zeros_like(image)
    cv.normalize(image, normalized, max, min, cv.NORM_MINMAX)
    if dtype is not None:
        normalized = normalized.astype(dtype)
    return normalized


def cv_norm(input: Union[Point, np.ndarray]) -> float:
    """
    Calculates the absolute L2 norm of the point or array.
    :param input: The n-dimensional point
    :return: The L2 norm of the n-dimensional point
    """
    if isinstance(input, Point):
        return cv.norm((*input,))
    else:
        return cv.norm(input)


def resize(
    image: np.ndarray,
    *,
    factor: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Resize an image with the given factor or shape.

    Either shape or factor must be provided.
    Using `factor` of 2 gives an image of half the size.
    Using `shape` gives an image of the given shape.

    :param image: Image to resize
    :param factor: Shrink factor. A factor of 2 halves the image size.
    :param shape: Output image size.
    :return: A resized image
    """
    _error_if_image_empty(image)
    if image.ndim == 2 or image.ndim == 3:
        if shape is not None:
            return cv.resize(image, shape, interpolation=cv.INTER_CUBIC)
        elif factor is not None:
            return cv.resize(
                image, None, fx=1 / factor, fy=1 / factor, interpolation=cv.INTER_CUBIC
            )
        else:
            raise ValueError("Either shape or factor must be specified.")
    raise ValueError("Image must have either 2 or 3 dimensions.")


def gray2bgr(image: np.ndarray) -> np.ndarray:
    """Convert image from gray to BGR

    :param image: Image to be converted
    :return: Converted image
    """
    _error_if_image_empty(image)
    _error_if_image_not_gray(image)
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def bgr2gray(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR to gray

    :param image: Image to be converted
    :return: Converted image
    """
    _error_if_image_empty(image)
    _error_if_image_not_color(image)
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def bgr2hsv(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR to HSV color space

    :param image: Image to be converted
    :return: Converted image
    """
    _error_if_image_empty(image)
    _error_if_image_not_color(image)
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def bgr2xyz(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR to CIE XYZ color space

    :param image: Image to be converted
    :return: Converted image
    """
    _error_if_image_empty(image)
    _error_if_image_not_color(image)
    return cv.cvtColor(image, cv.COLOR_BGR2XYZ)


def bgr2hls(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR to HLS color space

    :param image: Image to be converted
    :return: Converted image
    """
    _error_if_image_empty(image)
    _error_if_image_not_color(image)
    return cv.cvtColor(image, cv.COLOR_BGR2HLS)


def bgr2luv(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR to CIE LUV color space

    :param image: Image to be converted
    :return: Converted image
    """
    _error_if_image_empty(image)
    _error_if_image_not_color(image)
    return cv.cvtColor(image, cv.COLOR_BGR2LUV)


def blur_gaussian(
    image: np.ndarray, kernel_size: int = 3, sigma_x=None, sigma_y=None
) -> np.ndarray:

    _error_if_image_empty(image)

    if sigma_x is None:
        sigma_x = 0
    if sigma_y is None:
        sigma_y = 0

    return cv.GaussianBlur(
        image, ksize=(kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y
    )


def blur_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.medianBlur(image, kernel_size)


def threshold_adaptive(
    image: np.ndarray,
    block_size: int,
    c: int = 0,
    *,
    weighted: bool = True,
    inverse: bool = False,
) -> np.ndarray:
    """Adaptive thresholding of `image`, using a (weighted) local mean.

    A local threshold value is determined for each `block_size x block_size` window.
    If `weighted` is true, the gaussian weighted mean is used. If not, the mean is used.

    :param image: Input image.
    :param block_size: The size of the local windows.
    :param c: Constant to be subtracted from the (weighted) mean.
    :param weighted: Whether or not to weight the mean with a gaussian weighting.
    :param inverse: Whether or not to inverse the image output.
    :return: The thresholded image.
    """
    _error_if_image_empty(image)
    _error_if_image_wrong_dtype(image, [np.uint8])

    method = cv.ADAPTIVE_THRESH_GAUSSIAN_C if weighted else cv.ADAPTIVE_THRESH_MEAN_C
    flags = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY

    return cv.adaptiveThreshold(image, 255, method, flags, block_size, c)


def threshold_otsu(
    image: np.ndarray, max_value: int = 255, inverse: bool = False
) -> np.ndarray:
    _error_if_image_empty(image)
    _error_if_image_wrong_dtype(image, [np.float32, np.uint8])

    flags = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    flags += cv.THRESH_OTSU
    _, img = cv.threshold(image, 0, max_value, flags)
    return img


def threshold_binary(
    image: np.ndarray, value: int, max_value: int = 255, inverse: bool = False
) -> np.ndarray:
    _error_if_image_empty(image)
    _error_if_image_wrong_dtype(image, [np.float32, np.uint8])

    flags = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    _, img = cv.threshold(image, value, max_value, flags)
    return img


def threshold_tozero(
    image: np.ndarray, value: int, max_value: int = 255, inverse: bool = False
) -> np.ndarray:
    _error_if_image_empty(image)
    _error_if_image_wrong_dtype(image, [np.float32, np.uint8])

    flags = cv.THRESH_TOZERO_INV if inverse else cv.THRESH_TOZERO
    _, img = cv.threshold(image, value, max_value, flags)
    return img


def threshold_otsu_tozero(
    image: np.ndarray, max_value: int = 255, inverse: bool = False
) -> np.ndarray:
    _error_if_image_empty(image)
    _error_if_image_wrong_dtype(image, [np.float32, np.uint8])

    flags = cv.THRESH_TOZERO_INV if inverse else cv.THRESH_TOZERO
    flags += cv.THRESH_OTSU
    _, img = cv.threshold(image, 0, max_value, flags)
    return img


def canny(
    image: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    high_pass_size: int = 3,
    l2_gradient=True,
) -> np.ndarray:
    """
    Perform Canny's edge detection on `image`.

    :param image: The image to be processed.
    :param low_threshold: The lower threshold in the hysteresis thresholding.
    :param high_threshold: The higher threshold in the hysteresis thresholding.
    :param high_pass_size: The size of the Sobel filter, used to find gradients.
    :param l2_gradient: Whether to use the L2 gradient. The L1 gradient is used if false.
    :return: Binary image of thinned edges.
    """
    _error_if_image_empty(image)
    if high_pass_size not in [3, 5, 7]:
        raise ValueError(f"High pass size must be either 3, 5 or 7: {high_pass_size}")
    return cv.Canny(
        image,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=high_pass_size,
        L2gradient=l2_gradient,
    )


def rotate_image(
    image: np.ndarray, center: Point, angle: float, unit: AngleUnit = AngleUnit.RADIANS
) -> np.ndarray:
    """
    Rotate `image` `angle` degrees at `center`. `unit` specifies if
    `angle` is given in degrees or radians.

    :param image: The image to be rotated.
    :param center: The center of the rotation
    :param angle: The angle to be rotated
    :param unit: The unit of the angle
    :return: The rotated image.
    """
    _error_if_image_empty(image)

    if unit is AngleUnit.RADIANS:
        angle = 180 / np.pi * angle
    rotation_matrix = cv.getRotationMatrix2D((*center,), angle, scale=1)

    if image.ndim == 2:
        return cv.warpAffine(image, rotation_matrix, image.shape[::-1])
    elif image.ndim == 3:
        copy = np.zeros_like(image)
        shape = image.shape[-2::-1]  # The two first, reversed
        for i in range(copy.shape[-1]):
            copy[..., i] = cv.warpAffine(image[..., i], rotation_matrix, shape)
        return copy
    else:
        raise ValueError("Image must have 2 or 3 dimensions.")


def _error_if_image_wrong_dtype(image: np.ndarray, dtypes: Sequence[type]):
    if image.dtype not in dtypes:
        raise ValueError(
            f"Image wrong dtype {image.dtype}, expected one of {[dtype for dtype in dtypes]}"
        )


def _error_if_image_empty(image: np.ndarray) -> None:
    if image is None or len(image) == 0 or image.size == 0:
        raise ValueError("Image is empty")


def _error_if_image_not_color(image: np.ndarray) -> None:
    if image.ndim != 3:
        raise ValueError(f"Expected image with three channels: {image.ndim}")


def _error_if_image_not_gray(image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError(f"Expected image with two channels: {image.ndim}")
