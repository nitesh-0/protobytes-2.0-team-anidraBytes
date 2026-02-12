# Team anidraBytes - Drishtimarga

## Team Information
**Team Name:** anidraBytes

**Team Members:**
* **Bipin Kumar Marasini** - [Email] - [@Bipin-km](https://github.com/Bipin-km)
* **Dharmendra Singh Chaudhary** - [Email] - [@Dharmendra016](https://github.com/Dharmendra016)
* **Nitesh Kumar Sah** - [Email] - [@nitesh-0](https://github.com/nitesh-0)
* **Ramesh Kathayat** - [Email] - [@ramesh34-hub](https://github.com/ramesh34-hub)

---

## Project Details

**Project Title:** Drishtimarga

**Category:**
- [ ] FinTech
- [ ] EdTech
- [ ] E-Governance
- [ ] IoT
- [x] Open Innovation

**Problem Statement:**
[cite_start]Visually impaired individuals face significant safety risks from dynamic hazards like moving vehicles, yet existing navigation aids often lack the low latency (<300ms) required for real-time avoidance[cite: 37]. [cite_start]Current market solutions are either too expensive or fail to provide depth perception and velocity tracking in chaotic urban environments[cite: 35].

**Solution Overview:**
[cite_start]Drishtimarga is an AI-powered wearable "digital eye" that captures live video via an ESP32-CAM and processes it on a laptop GPU using a modular pipeline[cite: 36]. [cite_start]It combines object detection, monocular depth estimation, and spatial tracking to deliver clear, offline audio feedback about surroundings with sub-200ms latency[cite: 40].

---

## Technical Stack

**Frontend (Hardware & Capture):**
* [cite_start]ESP32-CAM (MJPEG Stream over WiFi) [cite: 39]
* [cite_start]OpenCV (Video Capture & Frame Management) [cite: 47]

**AI Models:**
* [cite_start]**Object Detection:** YOLOv8n (Nano) - *Selected for high speed (80-150 FPS) and accuracy* [cite: 124]
* [cite_start]**Depth Estimation:** Depth Anything V2 Small - *Selected for optimal monocular accuracy at ~30 FPS* [cite: 150]
* [cite_start]**Object Tracking:** ByteTrack - *Selected for persistent ID tracking and zero-setup integration* [cite: 191]
* [cite_start]**Spatial Mapping:** LocalSpatialMap (Hybrid Optical Flow) [cite: 39]

**Audio:**
* [cite_start]pyttsx3 - *Selected for offline, zero-latency text-to-speech* [cite: 278]

**Backend & Infrastructure:**
* Python 3.10+
* Ultralytics YOLO Library
* [cite_start]NVIDIA GPU (CUDA) for acceleration [cite: 39]

---

## Installation & Setup

[cite_start]Follow these steps to set up and run the project[cite: 643]:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/nitesh-0/protobytes-2.0-team-anidraBytes.git](https://github.com/nitesh-0/protobytes-2.0-team-anidraBytes.git)
    cd protobytes-2.0-team-anidraBytes
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run with webcam (simplest — starts immediately)**
    ```bash
    python main.py
    ```

4.  **Run with ESP32-CAM**
    ```bash
    # Ensure laptop is connected to ESP32 WiFi (AP mode)
    python main.py --source "[http://192.168.4.1:81/stream](http://192.168.4.1:81/stream)"
    ```

---

## Demo Credentials
*(Not Applicable - Hardware/Local Project)*

---

## Screenshots/Demo

![Drishtimarga Detection Demo](docs/demo_screenshot.png)
[cite_start]*(will be added soon)*