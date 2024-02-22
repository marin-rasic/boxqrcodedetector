import typing
import cv2 as cv
import numpy as np

from ultralytics import YOLO
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol


class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float):
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._conf = conf
        self._qr_code = ""

    def get_coordinates(self) -> typing.Tuple[int, int, int, int]:
        return self._x1, self._y1, self._x2, self._y2
    
    def get_confidence_score(self) -> float:
        return self._conf
    
    def upper_left_corner(self) -> typing.Tuple[int, int]:
        return self._x1, self._y1
    
    def lower_right_corner(self) -> typing.Tuple[int, int]:
        return self._x2, self._y2
    
    def set_qr_code(self, qr_code: str):
        self._qr_code = qr_code
    
    def get_qr_code(self) -> str:
        return self._qr_code
    

class BoxObjectDetector:
    def __init__(self, weights: str, image_width: int, image_height: int, conf: float):
        self._weights = weights
        self._model = YOLO(weights)
        self._conf = conf
        self._image_width = image_width
        self._image_height = image_height

    def is_right_image_size(self, image: np.ndarray) -> bool:
        height, width, _ = image.shape
        return height == self._image_height and width == self._image_width

    def resize_image(self, image: np.ndarray):
        dim = (self._image_width, self._image_height)
        return cv.resize(image, dim)

    def find_objects(self, image: np.ndarray) -> typing.List[BoundingBox]:
        if not self.is_right_image_size(image):
            image = self.resize_image(image)

        bounding_boxes = list()
        results = self._model(image, conf = self._conf)
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                bb = BoundingBox(int(x1), int(y1), int(x2), int(y2), float(box.conf))
                bounding_boxes.append(bb)

        return bounding_boxes
    

class QRCodeFinder:
    def __init__(self, format_function = None):
        self._qr_code_list = set()
        self._format_function = format_function

    def get_qr_codes(self) -> typing.Set[str]:
        return self._qr_code_list

    def clear_qr_codes(self):
        self._qr_code_list.clear()

    def image_binarization(self, image: np.ndarray):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        return thresh
    
    def check_qr_code_format(self, qr_code: str) -> bool:
        if self._format_function is None:
            return True
        
        return self._format_function(qr_code)

    def detect_and_decode_qr_code_in_box(self, image: np.ndarray, box: BoundingBox) -> str:
        x1, y1, x2, y2 = box.get_coordinates()
        crop_image = image[y1:y2, x1:x2].copy()
        binariazed_image = self.image_binarization(crop_image)
        decoded_info = decode(binariazed_image, symbols=[ZBarSymbol.QRCODE])
        if decoded_info:
            return decoded_info[0].data.decode('UTF-8') # we assume there will be only one QR code present
        
        return ""

    def find_qr_codes(self, image: np.ndarray, boxes: typing.List[BoundingBox]):
        for box in boxes:
            qr_code = self.detect_and_decode_qr_code_in_box(image, box)
            if qr_code and self.check_qr_code_format(qr_code):
                self._qr_code_list.add(qr_code)
                box.set_qr_code(qr_code)


