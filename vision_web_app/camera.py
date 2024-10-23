import asyncio
import cv2
import numpy as np
import logging
from jongphago.decorator.dec import Counter
from jongphago.logging.log import log_format, get_logger
from jongphago.vision.video import get_cheonahn_camera_url

logging.basicConfig(format=log_format, level=logging.ERROR, datefmt="%H:%M:%S")
camera_read_logger = get_logger("camera_read", "vision_web_app/log/camera_read.log")


class Camera:
    def __init__(self, url, out_queue):
        self.url = url
        self.name = url.split("/")[-1]
        self.out_queue = out_queue  # rtsp_queue
        self.capture = None
        self.frame = None
        self.is_opened = False
        self.size = (640, 380)

    async def initialize_capture(self):
        loop = asyncio.get_event_loop()
        self.capture = await loop.run_in_executor(None, cv2.VideoCapture, self.url)
        self.is_opened = self.capture.isOpened()

    async def read(self):
        await self.initialize_capture()
        while self.is_opened:
            self.frame = await self.get()
            if self.frame is None:
                break
            await self.put()

    async def run(self):
        await self.read()

    @Counter(logger=camera_read_logger)
    async def get(self) -> np.ndarray | None:
        loop = asyncio.get_event_loop()
        logging.info(
            f"Reading frame from {self.url.split('/')[-1]} | # tasks: {len(asyncio.all_tasks(loop))}"
        )
        ret, frame = await loop.run_in_executor(None, self.capture.read)
        if not ret:
            logging.error(f"Failed to read frame from {self.url}")
            return None
        return frame

    async def put(self):
        while self.out_queue.full():
            old_item = await self.out_queue.get()
            del old_item
        await self.out_queue.put((self.name, self.frame))


# 외부에서 Camera 클래스를 비동기적으로 실행
async def main():
    cameras = [
        Camera(get_cheonahn_camera_url(i), asyncio.Queue(maxsize=300))
        for i in range(6, 6 + 16)
    ]
    await asyncio.gather(*(camera.run() for camera in cameras))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
