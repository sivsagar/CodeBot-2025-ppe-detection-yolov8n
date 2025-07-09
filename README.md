# 🛡️ Smart PPE Detection System – Real-Time Safety Monitoring with AI

An intelligent, plug-and-play system built for real-time monitoring of Personal Protective Equipment (PPE) compliance using YOLOv8, Raspberry Pi, and live camera feeds. Designed for high-risk industrial environments where safety is paramount, this solution automatically detects violations and alerts supervisors through visual evidence and email notifications.

---

## 🚀 Why This Project?

Industrial and construction zones often face life-threatening incidents due to improper use of safety equipment. Manual monitoring is inefficient, error-prone, and expensive. This system provides an autonomous and scalable solution to ensure safety compliance 24/7 using AI and edge computing.

---

## 🔍 Key Features

- **Auto-Start on Boot** — Runs automatically on Raspberry Pi via `crontab`.
- **Real-Time Detection** — Streams live video and runs AI inferences on the edge device.
- **Snapshot Capture** — Saves images on PPE violation for audit trail.
- **Email Notification** — Sends alerts instantly with captured evidence.
- **Web Dashboard** — Displays live stream and statistics in a clean, modern UI.
- **Hardware Flexible** — Works with Pi Camera, Drone Camera, or Tripod-mounted USB cams.

---

## 🛠️ Tech Stack

| Layer              | Technology               |
|--------------------|--------------------------|
| AI Model           | YOLOv8n (Ultralytics)    |
| Backend            | Python + Flask           |
| Email Alerts       | SMTP (Gmail App Password)|
| Frontend           | HTML, CSS, JavaScript    |
| Camera Integration | Picamera2                |
| Edge Platform      | Raspberry Pi 5           |
| Auto Start         | Crontab                  |

---

## 📐 System Architecture

```mermaid
graph TD
A[Boot via Crontab] --> B[Initialize Pi Camera / Drone]
B --> C[Live Video Feed]
C --> D[YOLOv8n Inference]
D --> E{PPE Violation?}
E -- Yes --> F[Capture Snapshot]
F --> G[Send Email via SMTP]
E -- No --> H[Continue Monitoring]
F --> I[Show Stats on Web Dashboard]
H --> I
