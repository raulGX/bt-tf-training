import cv2
import sys
from PIL import Image
import requests
from io import BytesIO
import numpy as np


url = sys.argv[1]
response = requests.get(url)
im_str = BytesIO(response.content).read()
image = np.asarray(bytearray(im_str), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 100, 400)
cv2.imshow("im", edged)
cv2.waitKey(0)
_, cnts, _ = cv2.findContours(edged.copy(),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 50:
        idx += 1
        new_img = image[y:y+h, x:x+w]
        cv2.imshow("im", new_img)
        cv2.waitKey(0)
