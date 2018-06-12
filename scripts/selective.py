import selectivesearch
import cv2
import sys
from PIL import Image
import requests
from io import BytesIO
import numpy as np

url = sys.argv[1]
response = requests.get(url)
image = np.array(Image.open(BytesIO(response.content)))
img_lbl, regions = selectivesearch.selective_search(
    image, scale=500, sigma=0.7, min_size=4000)
for rect in regions:
    if (len(rect["labels"]) > 2):
        continue
    region = rect["rect"]
    (x1, y1, w, h) = region
    cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)

cv2.imshow("im", image)
cv2.waitKey(0)
