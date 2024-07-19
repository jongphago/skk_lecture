import sys
import logging

sys.path.append("vision_web_app")
import cv2
import numpy as np
from images.pts import ca_camera_7_pers_pts, ca_camera_7_top_pts

format = "%(asctime)s [%(process)d|%(thread)d](%(funcName)s:%(lineno)d): %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

pts_dict = {
    "perspective": ca_camera_7_pers_pts,
    "top": ca_camera_7_top_pts,
}

img_dict = {
    "perspective": "vision_web_app/images/ca_camera_7_pers.png",
    "top": "vision_web_app/images/ca_camera_7_top.png",
}

view = sys.argv[1]
pts = pts_dict[view]
img = cv2.imread(img_dict[view])
logging.debug(f"image size: {img.shape}")


for index, pt in enumerate(pts, 1):
    logging.debug(type(pt))
    cv2.circle(img, pt, 10, (0, 255, 0), -1)
    cv2.putText(
        img,
        str(index),
        (pt[0] - 20, pt[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
cv2.polylines(img, np.array([pts]), True, (0, 255, 0), 2)

cv2.imshow("Image", img)
wait_key = cv2.waitKey(0)
if wait_key == ord("q"):
    cv2.destroyAllWindows()
    sys.exit(0)
