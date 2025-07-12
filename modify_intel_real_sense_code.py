import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (use your PPE-trained model)
model = YOLO("yolov8s_custom.pt")  # Replace with your model path

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams at optimal settings
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Set High Accuracy visual preset
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3)  # 3 = High Accuracy

# Align depth to color
align = rs.align(rs.stream.color)

# Optional: apply filters to improve depth quality
decimation = rs.decimation_filter()
spatial    = rs.spatial_filter()
temporal   = rs.temporal_filter()

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Apply depth filters
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        # Convert color image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 model on the frame
        results = model(color_image)

        # Annotate results
        annotated_frame = results[0].plot()

        # Estimate and overlay distance
        if results[0].boxes:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = depth_frame.get_distance(cx, cy)
                cv2.putText(annotated_frame, f"{distance:.2f}m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Display result
        cv2.imshow("PPE Detection - RealSense + YOLOv8", annotated_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
