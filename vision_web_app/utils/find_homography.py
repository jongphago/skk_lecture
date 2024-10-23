import cv2
import numpy as np
from vision_web_app.images.pts import ca_camera_7_pers_pts, ca_camera_7_top_pts


def find_homography(cam_position, map_position, ransac_thresh=5.0, maxIters=100000):
    # RANSAC Candidates : cv2.USAC_ACCURATE, cv2.USAC_MAGSAC, cv2.RANSAC
    H, status = cv2.findHomography(
        cam_position,
        map_position,
        cv2.USAC_MAGSAC,
        ransac_thresh,
        maxIters=maxIters,
    )
    return H, status


if __name__ == "__main__":
    assert len(ca_camera_7_pers_pts) == len(ca_camera_7_top_pts)
    H, status = find_homography(
        np.array(ca_camera_7_pers_pts),
        np.array(ca_camera_7_top_pts),
    )
    print(f"ca_camera_7_top_pts = {ca_camera_7_top_pts}")
    print(f"ca_camera_7_pers_pts = {ca_camera_7_pers_pts}")
    print(f"ca_camera_7_homography = {H.tolist()}")
