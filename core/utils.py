import cv2
import numpy as np
import base64

class ImageUtils:
    @staticmethod
    def base64_to_cv2(b64_string: str):
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def check_orientation_qr(img):
        """Retorna True se estiver invertido, False se estiver correto"""
        qr_detector = cv2.QRCodeDetector()
        ok, points = qr_detector.detect(img)
        if ok and points is not None:
            qr_y = np.mean(points[0][:, 1])
            return qr_y > (img.shape[0] / 2)
        return False