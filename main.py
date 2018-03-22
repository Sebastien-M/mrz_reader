from mrz_detect.detect_mrz import MrzDetect
import argparse
import cv2


mrz_detect = MrzDetect()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
arg = vars(ap.parse_args())
mrz = mrz_detect.detect_mrz(arg)

cv2.imshow("mrz", mrz)
cv2.waitKey(0)
