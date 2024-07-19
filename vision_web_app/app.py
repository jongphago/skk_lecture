from collections import defaultdict
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils import cheonan_camera_7_homography as H
from utils import (
    get_track_results,
    plot_cctv_position,
    perspective_transform,
    get_cheonahn_camera_url,
)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")
names = model.model.names
track_history = defaultdict(lambda: [])
track_history_top = defaultdict(lambda: [])

# Open the RTSP stream
rtsp_url = get_cheonahn_camera_url(7)
cap = cv2.VideoCapture(rtsp_url)

# Load the CCTV data
cctv_csv = "vision_web_app/utils/cctv.csv"
cctv_df = pd.read_csv(cctv_csv)
fig = plot_cctv_position(cctv_df)
st.title("교통정보 CCTV")
st.plotly_chart(fig, use_container_width=True)

# Display the camera stream
st.title(rtsp_url.split("/")[-1])
st.write(f"camera url: {rtsp_url}")
col1, col2 = st.columns([1, 1])
frame_placeholder = col1.empty()
top_placeholder = col2.empty()
top_img_path = "vision_web_app/images/ca_camera_7_top.png"
_top_image = cv2.cvtColor(cv2.imread(top_img_path), cv2.COLOR_BGR2RGB)

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    top_image = _top_image.copy()

    # Perform object detection and tracking
    results = model.track(frame, persist=True, verbose=False)
    result = results[0]
    if result.boxes.id is None:
        continue

    # Initialize the Annotator
    annotator = Annotator(frame, line_width=2)
    annotator_top = Annotator(top_image, line_width=10)
    for box, cls, track_id, cxby in zip(*get_track_results(result)):
        # Initialize
        label = f"{names[int(cls)]} [{track_id}]"
        color = colors(int(cls), True)
        warped = perspective_transform(cxby, H)

        # Store tracking history
        track = track_history[track_id]
        track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
        if len(track) > 30:
            track.pop(0)

        # Store tracking history for top image
        track_top = track_history_top[track_id]
        track_top.append(warped)
        if len(track_top) > 30:
            track_top.pop(0)

        # Visualize the results on the frame (perspective)
        annotator.box_label(box, color=color, label=label)
        annotator.draw_centroid_and_tracks(track, color, 2)

        # Visualize the results on the top image (top)
        annotator_top.text(warped, label, color)
        annotator_top.draw_centroid_and_tracks(track_top, color, 10)

    # Display the frame and top image
    frame_placeholder.image(annotator.im)
    top_placeholder.image(annotator_top.im, use_column_width=True)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
