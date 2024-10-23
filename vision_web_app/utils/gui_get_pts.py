import cv2


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


if __name__ == "__main__":
    view = "perspective"[:4]
    image_path = f"vision_web_app/images/ca_camera_7_{view}.jpg"
    perspective_view_point_selector = PointSelector(image_path)
    pers_points = perspective_view_point_selector.get_points()

    view = "top"
    image_path = f"vision_web_app/images/_ca_camera_7_{view}.png"
    top_view_point_selector = PointSelector(image_path)
    top_points = top_view_point_selector.get_points()

    print("ca_camera_7_pers_pts = ", pers_points)
    print("ca_camera_7_top_pts = ", top_points)
