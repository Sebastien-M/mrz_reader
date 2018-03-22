from imutils import paths
import numpy as np
import imutils
import cv2


class MrzDetect:

    def __init__(self):
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 40))
        self.sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    def detect_mrz(self, args):
        for image_path in paths.list_images(args["images"]):
            image, gray = self._prepare_image(image_path)
            blackhat = self._find_dark_regions(gray)
            grad_x = self._reduce_false_mrz_detection(blackhat)
            thresh = self._threshold_image(grad_x)
            thresh = self._enclose_mrz_section(thresh)
            self._remove_borders(image, thresh)
            mrz = self._find_contours(thresh, gray, image)
            return mrz

    def _prepare_image(self, image_path):
        """
        Resizes the image and convert it to grayscale
        """
        image = cv2.imread(image_path)
        image = imutils.resize(image, height=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray

    def _find_dark_regions(self, gray):
        """
        Lowers image noise with gaussian blur
        Finds dark regions on light background
        """
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)
        return blackhat

    def _reduce_false_mrz_detection(self, blackhat):
        """
        ¯\_(ツ)_/¯
        """
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        return grad_x

    def _threshold_image(self, grad_x):
        """
        Automatically thresholds image with otsu's method
        """
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, self.rectKernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return thresh

    def _enclose_mrz_section(self, thresh):
        """Still must fix rectKernel value"""
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.sqKernel)
        thresh = cv2.erode(thresh, None, iterations=4)
        return thresh

    def _remove_borders(self, image, thresh):
        """
        Removes 5% of left and right image in case of borders are detected in mrz
        """
        p = int(image.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0

    def _find_contours(self, thresh, gray, image):
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:

            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            cr_width = w / float(gray.shape[1])

            if ar > 5 and cr_width > 0.75:
                pX = int((x + w) * 0.03)
                pY = int((y + h) * 0.03)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))

                roi = image[y:y + h, x:x + w].copy()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break

        return roi

