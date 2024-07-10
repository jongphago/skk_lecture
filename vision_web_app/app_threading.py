import os
import logging
from pathlib import Path
import queue
import cv2
import streamlit as st
import threading
import time
from utils import get_cheonahn_camera_url

format = "%(asctime)s [%(process)d|%(thread)d](%(funcName)s:%(lineno)d): %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def show_fps(frame, prev_time, frame_count, fps):
    curr_time = time.time()
    if curr_time - prev_time >= 1:
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.2f}", (8, 30), font, 0.7, (0, 255, 0), 2)
    return frame, (prev_time, frame_count + 1, fps)


def stream_all_urls(urls, queues, target):
    logging.debug(urls, queues)
    params = [(url, q) for url, q in zip(urls, queues)]
    logging.debug(len(params))
    return [
        threading.Thread(target=target, args=param, daemon=True) for param in params
    ]


def read_frame(url, frame_queue):
    def _read(capture):
        return capture.read()

    capture = cv2.VideoCapture(url)
    name = Path(url).name
    ret = True
    args = [time.time(), 0, 0]  # prev_time, frame_count, fps
    logging.info(f"Read frame: {name}")
    while ret:
        ret, frame = _read(capture)
        frame = cv2.resize(frame, (640, 380))
        frame, args = show_fps(frame, *args)
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()  # discard previous (unprocessed) frame
            except queue.Empty:
                pass
        frame_queue.put((os.getpid(), name, frame))
        time.sleep(0.01)


def _write(name, frame, placeholder):
    placeholder.image(frame, channels="BGR")
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        exit(1)


def write_frame(queues, placeholders):
    for frame_queue, placeholder in zip(queues, placeholders):
        if frame_queue.empty():
            logging.debug(f"Empty queue: {id(frame_queue)}")
            time.sleep(0.01)
        _, name, frame = frame_queue.get()
        _write(name, frame, placeholder)


def main(urls, frame_placeholders):
    queues = [queue.Queue() for _ in (range(NUM_CAMERAS))]
    thread_list = stream_all_urls(urls, queues, read_frame)
    logging.debug(thread_list)
    for thread in thread_list:
        logging.info(f"Thread {thread.name} started")
        thread.start()

    logging.debug("Begin write frames")
    while True:
        write_frame(queues, frame_placeholders)


# Streamlit app
if __name__ == "__main__":
    logging.info("Begin streamlit app")
    NUM_CAMERAS = 2
    camera_index = 6
    rtsp_urls = [
        get_cheonahn_camera_url(i)
        for i in range(camera_index, camera_index + NUM_CAMERAS)
    ]
    logging.info(rtsp_urls)
    cols = st.columns((1, 1))
    frame_placeholders = []
    for rtsp_url, col in zip(rtsp_urls, cols):
        col.title(rtsp_url.split("/")[-1])
        col.write(f"camera url: {rtsp_url}")
        frame_placeholder = col.empty()
        frame_placeholders.append(frame_placeholder)

    main(rtsp_urls, frame_placeholders)
