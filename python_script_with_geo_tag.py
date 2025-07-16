from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import serial
import pynmea2
from email.message import EmailMessage
import smtplib
from picamera2 import Picamera2
from libcamera import controls

app = Flask(__name__, template_folder='templates')

# Global counters
ppe_on_count = 0
ppe_off_count = 0

# Setup camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()

# Load model
model = YOLO("yolov8s_custom.pt")

# Email credentials
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"

# Snapshot folder
os.makedirs("snapshots", exist_ok=True)

# Setup GPS Serial
gps_serial = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

def get_gps_coords():
    try:
        line = gps_serial.readline().decode('ascii', errors='replace')
        if line.startswith('$GPGGA'):
            msg = pynmea2.parse(line)
            lat = msg.latitude
            lon = msg.longitude
            return lat, lon
    except Exception as e:
        print("GPS error:", e)
    return None, None

def send_email_alert(snapshot_path, lat=None, lon=None):
    msg = EmailMessage()
    msg["Subject"] = "PPE Violation Detected with GPS"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECEIVER

    content = "A PPE violation was detected.\n"
    if lat and lon:
        content += f"Location: Latitude {lat}, Longitude {lon}\n"
        content += f"Google Maps Link: https://maps.google.com/?q={lat},{lon}\n"
    else:
        content += "GPS location not available.\n"

    msg.set_content(content)

    with open(snapshot_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(snapshot_path))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("📧 Email sent with GPS tag!")
    except Exception as e:
        print("❌ Email failed:", e)

def detect_ppe():
    global ppe_on_count, ppe_off_count

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        results = model(frame, imgsz=640, verbose=False)[0]
        ppe_on, ppe_off = 0, 0

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                ppe_on += 1
            else:
                ppe_off += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = f"snapshots/violation_{timestamp}.jpg"
                cv2.imwrite(snapshot_path, frame)
                lat, lon = get_gps_coords()
                send_email_alert(snapshot_path, lat, lon)

        ppe_on_count = ppe_on
        ppe_off_count = ppe_off

        annotated = results.plot()
        _, jpeg = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(detect_ppe(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/ppe-stats")
def ppe_stats():
    return jsonify({"ppe_on": ppe_on_count, "ppe_off": ppe_off_count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
