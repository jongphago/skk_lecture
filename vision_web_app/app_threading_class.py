import gc
import time
import queue
import logging
import threading
from itertools import repeat
import cv2
import requests
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils import timer, get_cheonahn_camera_url, show_fps

format = "%(asctime)s [%(process)d|%(thread)d](%(funcName)s:%(lineno)d): %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


class Camera(threading.Thread):
    def __init__(
        self,
        url,
        out_queue,
        daemon=True,
    ):
        super().__init__()
        self.url = url
        self.name = url.split("/")[-1]
        self.out_queue = out_queue  # rtsp_queue
        self.capture = None
        self.frame = None
        self.daemon = daemon
        self.is_opened = False
        self.size = (640, 380)
        self.args = [time.time(), 0, 0]  # prev_time, frame_count, fps

    def initialize_capture(self):
        self.capture = cv2.VideoCapture(self.url)
        self.is_opened = self.capture.isOpened()

    def run(self):
        logging.debug("Camera thread started")
        self.read()
        logging.debug("Camera thread finished")

    def read(self):
        self.initialize_capture()
        while self.is_opened:
            self.frame = self.get()
            if self.frame is None:
                break
            self.put()

    def get(self) -> np.ndarray | None:
        ret, frame = self.capture.read()
        if not ret:
            logging.error(f"Failed to read frame from {self.url}")
            return None
        return frame

    def put(self):
        if self.out_queue.full():
            old_item = self.out_queue.get()
            del old_item
            gc.collect()
        self.out_queue.put((self.name, self.frame))


class Visualizer(threading.Thread):
    def __init__(
        self,
        in_queue,
        out_queue,
        size=(640, 380),
        daemon=True,
    ):
        super().__init__()
        assert isinstance(in_queue, queue.Queue)
        assert isinstance(out_queue, queue.Queue)
        self.name = "Visualizer"
        self.in_queue = in_queue  # rtsp_queue
        self.out_queue = out_queue  # vis_queue
        self.daemon = daemon
        self.names = YOLO().model.names  # class names
        self.args = [time.time(), 0, 0]  # prev_time, frame_count, fps
        self.frame = None
        self.size = size

    def run(self):
        logging.debug("Camera thread started")
        self.visualize()
        logging.debug("Camera thread finished")

    def visualize(self):
        while True:
            result, self.frame = self.get()
            if self.frame is None:
                time.sleep(0.01)
                continue
            self.detect_postprocess(result)
            self.rtsp_postprocess()
            self.put()

    def rtsp_postprocess(self):
        self.frame = cv2.resize(self.frame, self.size)
        self.frame, self.args = show_fps(self.frame, *self.args)

    def detect_postprocess(self, results):
        annotator = Annotator(self.frame)
        boxes = [[int(i) for i in list(result["box"].values())] for result in results]
        clss = [result["class"] for result in results]
        for box, cls in zip(boxes, clss):
            label = f"{self.names[int(cls)]}"
            color = colors(int(cls), True)
            annotator.box_label(box, color=color, label=label)
        self.frame = annotator.im

    def get(self) -> np.ndarray | None:
        if self.in_queue.empty():
            logging.debug(f"Empty queue: {self.name}")
            return None, None
        self.name, result, frame = self.in_queue.get()
        logging.debug(result)
        return result, frame

    def put(self):
        if self.out_queue.full():
            old_item = self.out_queue.get()
            del old_item
            gc.collect()
        self.out_queue.put((self.name, self.frame))


class Collector(threading.Thread):
    def __init__(self, in_queues, out_queue, num_batch=16, daemon=True):
        super().__init__()
        assert isinstance(in_queues, dict)
        assert isinstance(out_queue, queue.Queue)
        self.name = "Collector"
        self.frame = None
        self.in_queues = in_queues
        self.out_queue = out_queue
        self.num_batch = num_batch
        self.daemon = daemon
        self.batches = []

    def run(self):
        logging.debug("Collector thread started")
        self.collect()
        logging.debug("Collector thread finished")

    def collect(self):
        repeats = repeat(self.in_queues)
        for in_queues in repeats:  # temp
            for self.name, in_queue in in_queues.items():
                self.frame = self.get(in_queue)
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                self.put()

    def get(self, in_queue) -> np.ndarray | None:
        if in_queue.empty():
            logging.debug(f"Empty queue: {id(in_queue)}")
            return None
        _, frame = in_queue.get()
        logging.debug(f"{self.name}: {frame.shape}")
        return frame

    def put(self):
        self.batches.append((self.name, self.frame))
        if len(self.batches) == self.num_batch:
            if self.out_queue.full():
                old_item = self.out_queue.get()
                del old_item
                gc.collect()
            self.out_queue.put(self.batches)
            self.batches = []


class Detecter(threading.Thread):
    def __init__(
        self,
        in_queues: dict[str : queue.Queue],
        out_queues: dict[str : queue.Queue],
        checkpoint="yolov8n.pt",
    ):
        super().__init__()
        assert isinstance(in_queues, dict)
        assert isinstance(out_queues, dict)
        self.model = YOLO(checkpoint)
        self.model_name = self.model.model.names
        self.name = "Detecter"
        self.names: list[str] = None
        self.frames: list[np.ndarray] = None
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.result = None

        self.collect_queue = queue.Queue(maxsize=100)
        self.collector = Collector(self.in_queues, self.collect_queue)

    def run(self):
        logging.debug("Detecter thread started")
        self.detect()
        logging.debug("Detecter thread finished")

    def run_collector(self):
        self.collector.start()
        logging.debug("Collector thread started")

    def detect(self):
        self.run_collector()
        while True:
            self.frames = self.get()
            if self.frames is None:
                time.sleep(0.01)
                continue
            if not self.inference():
                time.sleep(0.01)
                continue
            self.put()

    def inference(self):
        response = send(self.frames)
        if response.status_code != 200:
            logging.error(f"Failed to send frames to server: {response.status_code}")
            return False
        self.results = response.json()["batch"]
        # self.results = self.model(self.frames, persist=True, verbose=False)
        # self.results = self.model.detect(self.frames, persist=True, verbose=False)
        # self.results = [0] * len(self.frames)  # temp
        return True

    def get(self):
        if self.collect_queue.empty():
            logging.debug(f"Empty queue: {self.name}")
            return None
        batch = self.collect_queue.get()
        self.names, frames = [], []
        for name, frame in batch:
            self.names.append(name)
            frames.append(frame)
        return frames

    def put(self):
        for name, result, frame in zip(self.names, self.results, self.frames):
            if self.out_queues[name].full():
                _ = self.out_queues[name].get()
            self.out_queues[name].put((name, result, frame))


class Model(threading.Thread):
    def __init__(self):
        super().__init__()
        self.name = "Model"
        self.model = YOLO()
        self.names = self.model.names

    def run(self):
        logging.debug("Model thread started")
        self.inference()
        logging.debug("Model thread finished")

    def inference(self):
        raise NotImplementedError


@timer
def send(
    frames: list[np.ndarray],
    url: str = "http://34.64.235.71:8000/image_files/",
):
    def array2bytes(frame: np.ndarray, format: str = ".jpg") -> bytes:
        _, encoded = cv2.imencode(format, frame)
        return encoded.tobytes()

    images_bytes = [array2bytes(frame) for frame in frames]
    files = [
        ("images", (f"image_{i}.jpg", image_bytes, "image/jpeg"))
        for i, image_bytes in enumerate(images_bytes)
    ]
    response = requests.post(url, files=files)
    return response


class App:
    def __init__(self, urls):
        def get_cctv_key(url):
            return url.split("/")[-1].split(".")[0]

        self.urls = urls
        self.rtsp_queues = {get_cctv_key(url): queue.Queue(maxsize=100) for url in urls}
        self.detect_queues = {
            get_cctv_key(url): queue.Queue(maxsize=100) for url in urls
        }
        self.vis_queues = {get_cctv_key(url): queue.Queue(maxsize=100) for url in urls}

        cam_out_queues = self.rtsp_queues
        detect_in_queues = self.rtsp_queues
        detect_out_queues = self.detect_queues
        vis_in_queues = self.detect_queues
        vis_out_queues = self.vis_queues

        self.cameras = [
            Camera(url, out_queue)
            for url, out_queue in zip(self.urls, cam_out_queues.values())
        ]
        self.detecter = Detecter(detect_in_queues, detect_out_queues)
        self.visualizers = [
            Visualizer(in_queue, out_queue)
            for in_queue, out_queue in zip(
                vis_in_queues.values(), vis_out_queues.values()
            )
        ]

    def start(self):
        for camera in self.cameras:
            camera.start()
        logging.info("Camera thread started")

        self.detecter.start()
        logging.info("Detecter thread started")

        for visualizer in self.visualizers:
            visualizer.start()
        logging.info("Visualizer thread started")

    def join(self):
        for camera in self.cameras:
            camera.join()
        logging.info("App thread joined")

        for visualizer in self.visualizers:
            visualizer.join()
        logging.info("Visualizer thread joined")

    def show(self):
        st.title("CCTV")
        cols = st.columns(len(self.urls))
        frame_placeholders = [col.empty() for col in cols]
        while True:
            for (
                index,
                visualizer,
            ) in enumerate(
                self.visualizers,
            ):
                frame = self.get(visualizer.out_queue)
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_placeholders[index].image(frame, channels="BGR")
                # cv2.imshow(f"{index}", frame)
                # if cv2.waitKey(1) == ord("q"):
                #     cv2.destroyAllWindows()
                #     exit(1)

    def get(self, queue) -> np.ndarray | None:
        if queue.empty():
            logging.debug("Empty queue")
            return None
        _, frame = queue.get()  # name, frame
        return frame

    def run(self):
        self.start()
        self.show()
        self.join()
        logging.info("App thread finished")
        alived = threading.enumerate()
        logging.info(f"Alived threads: {alived}")


@timer
def main():
    urls = [get_cheonahn_camera_url(i) for i in range(6, 8)]
    app = App(urls)
    app.run()


if __name__ == "__main__":
    main()
