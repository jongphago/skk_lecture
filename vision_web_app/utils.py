import time
import logging
from functools import wraps
import cv2
import numpy as np
import plotly.express as px


def get_cheonahn_camera_url(camera_index: int) -> str:
    url = f"rtsp://210.99.70.120:1935/live/cctv{camera_index:03d}.stream"
    return url


def perspective_transform(bm, homography):
    bmx, bmy = bm  # bottom-mid
    cam_point = np.array([[[bmx, bmy]]], dtype=np.float32)
    map_point = cv2.perspectiveTransform(cam_point, homography)
    world_x, world_y = map_point[0][0]
    return (int(world_x), int(world_y))


cheonan_camera_7_homography = np.array(
    [
        [0.45088523457483415, -4.253659372434125, 1295.9824904107918],
        [0.12610650973964696, 0.24372319507984377, -985.1596091467705],
        [7.455742049638113e-06, -0.0028470223490374763, 0.9999999999999999],
    ]
)


def plot_cctv_position(cctv_df, zoom=12, width=400, mapbox_style="carto-positron"):
    # Plotly Mapbox를 사용하여 지도 표시
    fig = px.scatter_mapbox(
        cctv_df,
        lat="위도",
        lon="경도",
        hover_name="설치위치명",
        hover_data=["CCTV관리번호", "설치위치주소"],
        zoom=zoom,
        width=width,
    )
    # Mapbox 스타일 설정
    fig.update_layout(mapbox_style=mapbox_style)
    return fig


def get_track_results(result):
    def xyxy2cxby(xyxy) -> np.ndarray:
        cx = (xyxy[:, 0] + [xyxy[:, 2]]) // 2
        cx = cx.reshape(-1, 1)
        by = np.expand_dims(xyxy[:, 3], axis=1)
        cxby = np.concatenate([cx, by], axis=1)
        return cxby

    boxes = result.boxes.xyxy.cpu()
    clss = result.boxes.cls.numpy().astype(int)
    track_ids = result.boxes.id.numpy().astype(int)
    _xyxy = result.boxes.xyxy.numpy().astype(int)
    cxbys = xyxy2cxby(_xyxy[:])  # center x, bottom y
    return boxes, clss, track_ids, cxbys


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.debug(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper


def show_fps(frame, prev_time, frame_count, fps):
    curr_time = time.time()
    if curr_time - prev_time >= 1:
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.2f}", (8, 30), font, 0.7, (0, 255, 0), 2)
    return frame, (prev_time, frame_count + 1, fps)


def read_batch_frame(self):
    frames: np.ndarray = None
    for _ in range(self.num_batch):
        ret, frame = self._read_single_frame()
        if not ret:
            break
        frame = np.resize(frame, self.shape)
        frame = np.expand_dims(frame, axis=0)
        if frames is not None:
            frames = np.concatenate((frames, frame), axis=0)
        else:
            frames = frame
    return frames
