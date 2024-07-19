import sys
import logging

sys.path.append("vision_web_app")
import cv2
import numpy as np
from images.pts import (
    ca_camera_7_pers_pts,
    ca_camera_7_top_pts,
    ca_camera_7_homography,
)

# Set the logging format and level
format = "%(asctime)s [%(process)d|%(thread)d](%(funcName)s:%(lineno)d): %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

# Define the points and the image
pts_dict = {
    "perspective": ca_camera_7_pers_pts,
    "top": ca_camera_7_top_pts,
}
img_dict = {
    "perspective": "vision_web_app/images/ca_camera_7_pers.png",
    "top": "vision_web_app/images/ca_camera_7_top.png",
}

# Load the image and the points
img = cv2.imread(img_dict["perspective"])
logging.debug(f"image size: {img.shape}")
pts = pts_dict["perspective"]
homography = np.array(ca_camera_7_homography)

# Warp the image
warped = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

# Warp the points and draw pts, text, polyline on the warped image
pl_list = []
for index, pt in enumerate(pts, 1):
    ptx, pty = pt  # bottom-mid
    cam_point = np.array([[[ptx, pty]]], dtype=np.float32)
    warped_array = cv2.perspectiveTransform(cam_point, homography)[0][0]
    warped_pt = warped_array.astype(int).tolist()
    pl_list.append(warped_pt)
    logging.debug(type(warped_pt))
    cv2.circle(warped, warped_pt, 10, (0, 255, 0), -1)
    cv2.putText(
        warped,
        str(index),
        (warped_pt[0] - 20, warped_pt[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
cv2.polylines(warped, np.array([pl_list]), True, (0, 255, 0), 2)

# Display the warped image
cv2.imshow("Image", warped)
wait_key = cv2.waitKey(0)
if wait_key == ord("q"):
    cv2.destroyAllWindows()
    sys.exit(0)
