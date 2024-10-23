import cv2
import numpy as np
import matplotlib.pyplot as plt


class PointSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []
        self.current_point = None
        self.load_image()

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        cv2.imshow("Image", self.image)
        cv2.setMouseCallback("Image", self.mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_point = (x, y)
            cv2.circle(self.image, (x, y), radius=10, color=(0, 255, 0), thickness=-1)
            cv2.imshow("Image", self.image)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.current_point:
                self.points.append(self.current_point)
                self.current_point = None

    def get_points(self):
        return self.points


class HomographyFinder:
    def __init__(self, cam_path, map_path):
        self.cam_path = cam_path
        self.map_path = map_path

    def preprocess_images(slef):
        """카메라 이미지(cam_image)와 평면 이미지(map_image)의 크기 차이가 많이 나는 경우 올바른 변환 행렬(H)을 찾을 수 없다.
        이 경우 이미지를 resize하여 두 이미지의 크기를 비슷하게 맞춘다.
        """
        raise NotImplementedError

    def initialize_images(self):
        cam_image = self.read_and_convert(self.cam_path)
        map_image = self.read_and_convert(self.map_path)
        images = cam_image, map_image
        return images

    def find_homography(self):
        cam_points = np.array(self.cam_points)
        map_points = np.array(self.map_points)
        H, _ = cv2.findHomography(cam_points, map_points)
        return H

    def warp_cam_image(self):
        cam_image = self.draw_points(self.cam_image, self.cam_points)
        h, w = cam_image.shape[:2]
        warped_image = cv2.warpPerspective(
            cam_image,
            self.H,
            (w, h),
        )
        return warped_image

    def draw_points(self, image, points):
        drawn = cv2.polylines(
            image,
            [np.array(points)],
            True,
            (255, 0, 0),
            2,
        )
        return drawn

    def visualize_results(self):
        cam_image = self.draw_points(self.cam_image, self.cam_points)
        map_image = self.draw_points(self.map_image, self.map_points)
        warped_image = self.warped_image

        _, axes = plt.subplots(1, 3)
        axes[0].imshow(cam_image)
        axes[1].imshow(map_image)
        axes[2].imshow(warped_image)

        plt.show()

    def find(self):
        # Initialize the images
        images = self.initialize_images()
        self.cam_image, self.map_image = images
        # Get the points
        self.cam_points = self.get_points(self.cam_path)
        self.map_points = self.get_points(self.map_path)
        # Find the homography
        self.H = self.find_homography()
        # Warp the image
        self.warped_image = self.warp_cam_image()
        # Visualize the points
        self.visualize_results()

    def __call__(self):
        self.find()

    @staticmethod
    def read_and_convert(path):
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def get_points(path):
        selector = PointSelector(path)
        return selector.get_points()


if __name__ == "__main__":

    # Define the paths
    view = "perspective"[:4]
    cam_path = f"vision_web_app/images/ca_camera_7_{view}.jpg"
    view = "top"
    map_path = f"vision_web_app/images/_ca_camera_7_{view}.png"

    homography_finder = HomographyFinder(cam_path, map_path)
    homography_finder()

    # # Visualize the images
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(cv2.imread(map_path), cv2.COLOR_BGR2RGB))
    # plt.show()

    # # Get the points
    # perspective_view_point_selector = PointSelector(cam_path)
    # top_view_point_selector = PointSelector(map_path)
    # ca_camera_7_pers_pts = perspective_view_point_selector.get_points()
    # ca_camera_7_top_pts = top_view_point_selector.get_points()

    # # Find the homography
    # H, _ = cv2.findHomography(
    #     np.array(ca_camera_7_pers_pts),
    #     np.array(ca_camera_7_top_pts),
    # )

    # # Visualize the points
    # cam_image = cv2.imread(cam_path)
    # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    # cam_image = cv2.polylines(
    #     cam_image, [np.array(ca_camera_7_pers_pts)], True, (255, 0, 0), 2
    # )

    # map_image = cv2.imread(map_path)
    # map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
    # map_image = cv2.polylines(
    #     map_image, [np.array(ca_camera_7_top_pts)], True, (255, 0, 0), 2
    # )

    # h, w = cam_image.shape[:2]  # 변환할 이미지의 너비와 높이 추출
    # warped_image = cv2.warpPerspective(
    #     cam_image, np.array(H), (w, h)
    # )  # frame에 homography 적용

    # # Visualize the images
    # fig, axes = plt.subplots(1, 3)
    # axes[0].imshow(cam_image)
    # axes[1].imshow(map_image)
    # axes[2].imshow(warped_image)
    # plt.show()
