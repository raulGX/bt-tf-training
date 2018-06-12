import cv2
import sys
from PIL import Image
import requests
from io import BytesIO
import numpy as np


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50:
                return True
            elif i == row1-1 and j == row2-1:
                return False


url = sys.argv[1]
response = requests.get(url)
im_str = BytesIO(response.content).read()
image = np.asarray(bytearray(im_str), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 250, 500)
cv2.imshow("im", edged)
cv2.waitKey(0)

_, cnts, _ = cv2.findContours(edged.copy(),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
boxes = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 30 and h > 30:
        # idx += 1
        boxes.append(c)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

LENGTH = len(boxes)
status = np.zeros((LENGTH, 1))

for i, cnt1 in enumerate(boxes):
    x = i
    if i != LENGTH-1:
        for j, cnt2 in enumerate(boxes[i+1:]):
            x = x+1
            dist = find_if_close(cnt1, cnt2)
            if dist == True:
                val = min(status[i], status[x])
                status[x] = status[i] = val
            else:
                if status[x] == status[i]:
                    status[x] = i+1

unified = []
maximum = int(status.max())+1
for i in range(maximum):
    pos = np.where(status == i)[0]
    if pos.size != 0:
        cont = np.vstack(boxes[i] for i in pos)
        hull = cv2.convexHull(cont)
        unified.append(hull)

cv2.drawContours(image, unified, -1, (0, 255, 0), 2)
cv2.imshow("im", image)
cv2.waitKey(0)
