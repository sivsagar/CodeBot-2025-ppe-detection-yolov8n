import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (use your PPE-trained model here)
model = YOLO("yolov8s_custom.pt")  # Replace with "yolov8n.pt" or your own .pt file

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    while True:
        # Capture frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert RealSense image to NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 on the image
        results = model(color_image)

        # Plot detection results
        annotated_frame = results[0].plot()

        # Optional: get distance to first detected object (center of box)
        if results[0].boxes:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = depth_frame.get_distance(cx, cy)
                cv2.putText(annotated_frame, f"{distance:.2f}m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Show the result
        cv2.imshow("PPE Detection - RealSense + YOLOv8", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
