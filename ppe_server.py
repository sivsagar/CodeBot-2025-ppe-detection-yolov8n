from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
from email.message import EmailMessage
import smtplib
from picamera2 import Picamera2
from libcamera import controls

app = Flask(__name__, static_url_path='', static_folder='static')

# Global counters
ppe_on_count = 0
ppe_off_count = 0

# Setup camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()

# Load YOLOv8 model
model = YOLO("yolov8s_custom.pt")  # Replace with your trained model path

# Email config
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # App password from Google
EMAIL_RECEIVER = "your_email@gmail.com"

# Create snapshot folder
os.makedirs("snapshots", exist_ok=True)

def send_email_alert(snapshot_path):
    msg = EmailMessage()
    msg["Subject"] = "⚠️ PPE Violation Detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("A person was detected without full PPE kit. Snapshot attached.")

    with open(snapshot_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(snapshot_path))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("📧 Email sent with snapshot!")
    except Exception as e:
        print("❌ Email failed:", e)

def detect():
    global ppe_on_count, ppe_off_count
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        results = model(frame, imgsz=640, verbose=False)[0]
        ppe_on = 0
        ppe_off = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # PPE ON
                ppe_on += 1
            else:  # PPE OFF
                ppe_off += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"snapshots/violation_{timestamp}.jpg"
                cv2.imwrite(path, frame)
                send_email_alert(path)

        ppe_on_count = ppe_on
        ppe_off_count = ppe_off

        annotated = results.plot()
        _, jpeg = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(detect(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/ppe-stats")
def stats():
    return jsonify({"ppe_on": ppe_on_count, "ppe_off": ppe_off_count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
