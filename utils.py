import numpy as np
import cv2
import base64
from .option import Option

@Option.wrap
def base64_to_image(uri:str)->Option:
    """
    Converts a base64 image to an numpy array image
    """

    split_uri = uri.split(',')
    encoded_data = split_uri[0] if len(split_uri) == 1 else split_uri[0]
    array = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img: np.ndarray) -> bytes:
    """
    Converts a numpy image to base 64
    """

    img_buffer = cv2.imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')

