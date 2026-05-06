# IOT-EYECORE-ObjectDetection




# EYECORE — Neural Vision System

Real-time object detection dashboard using YOLOv8 and an ESP32-CAM as the video source.

---

## Requirements

- Python 3.10 or newer
- Arduino IDE 2.x
- ESP32-CAM (AI Thinker model)
- FTDI USB-to-Serial adapter (3.3V)

---

## Part 1 — ESP32-CAM Setup

### Step 1: Install Arduino IDE and ESP32 board package

Download Arduino IDE 2.x from https://arduino.cc

Then go to File → Preferences and add this URL under "Additional boards manager URLs":

    https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

Then go to Tools → Board → Boards Manager, search "esp32" by Espressif, and install it.

---

### Step 2: Install the esp32cam library

Go to Tools → Manage Libraries, search for "esp32cam" and install the one by yoursunny.

This provides WifiCam.hpp and the streamMjpeg() function used in the sketch.

If not found in the library manager, install manually from:
https://github.com/yoursunny/esp32cam

---

### Step 3: Select the correct board

Go to Tools → Board → esp32 and select:

    AI Thinker ESP32-CAM

Also set:
- Tools → Upload Speed → 115200
- Tools → Port → (your FTDI COM port)

---

### Step 4: Wire the FTDI adapter

The ESP32-CAM has no USB port. Use an FTDI USB-to-Serial adapter (3.3V) to upload.

    FTDI          ESP32-CAM
    --------      ---------------
    GND      →    GND
    VCC (5V) →    5V
    TX       →    U0R (GPIO3)
    RX       →    U0T (GPIO1)
    GND      →    IO0  ← upload mode only, remove after flashing

NOTE: IO0 connected to GND puts the board into flash/upload mode.
Remove this wire after uploading is done, then press RST to run normally.

---

### Step 5: Edit WiFi credentials

Open the sketch and update these two lines at the top:

    static const char* WIFI_SSID = "YOUR_WIFI_NAME";
    static const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

NOTE: ESP32-CAM only supports 2.4 GHz WiFi. 5 GHz networks will not connect.

---

### Step 6: Upload the sketch

1. Connect IO0 to GND (flash mode)
2. Click Upload in Arduino IDE
3. When you see "Connecting..." in the console, press the RST button on the board once
4. Wait for upload to finish
5. Remove the IO0 → GND wire
6. Press RST again to boot normally

Open Serial Monitor at 115200 baud. You will see the camera's IP address:

    http://192.168.x.x

You can verify the stream is working by visiting:

    http://192.168.x.x/resolutions.csv        (lists supported resolutions)
    http://192.168.x.x/1024x768.mjpeg         (live MJPEG stream)

---

## Part 2 — Python (EYECORE) Setup

### Step 1: Check Python version

    python --version

Must be 3.10 or newer.

---

### Step 2: Create a virtual environment (recommended)

    python -m venv eyecore-env

    # Windows
    eyecore-env\Scripts\activate

    # macOS / Linux
    source eyecore-env/bin/activate

---

### Step 3: Install dependencies

    pip install flask opencv-python numpy requests ultralytics

NOTE: ultralytics installs PyTorch automatically. This may take a few minutes
and requires around 1-2 GB of disk space. On first run, it will also download
the YOLOv8 nano model file (yolov8n.pt, ~6 MB) automatically.

---

### Step 4: Save the script

Save the full Python code as eyecore.py in your project folder.

---

### Step 5: Update the stream URL

Open eyecore.py and find this line near the top:

    STREAM_URL = "http://10.181.135.37/1280x1024.mjpeg"

Replace the IP address with the one shown in your ESP32-CAM Serial Monitor output.
Use a resolution that your camera supports (check /resolutions.csv first).

Example:

    STREAM_URL = "http://192.168.1.50/1024x768.mjpeg"

---

### Step 6: Run the app

    python eyecore.py

Then open your browser and go to:

    http://localhost:5000

Other devices on the same network can access it via your machine's IP:

    http://192.168.x.x:5000

---

## Troubleshooting

### ESP32-CAM

    "A fatal error: Failed to connect"
    → IO0 not connected to GND, or you missed pressing RST at the right moment.
      Try again: connect IO0→GND, click Upload, press RST when "Connecting..." appears.

    "Camera initialize failure"
    → Wrong board selected in Arduino IDE, or insufficient power.
      Try a dedicated 5V power supply instead of USB — the ESP32-CAM draws more
      current than some USB ports can supply.

    Blank or no stream in browser
    → Confirm the IP in Serial Monitor and test the URL directly in your browser first.

### Python / EYECORE

    "cv2 not found"
    → Try: pip install opencv-python-headless
      (use this on servers or systems without a display)

    "torch not found"
    → Run: pip install torch  (then re-run the full pip install command)

    Port 5000 already in use
    → Change the last line of eyecore.py from port=5000 to port=5001 (or any free port)

    Camera stream not connecting
    → Test with: ping 192.168.x.x
      Make sure the ESP32-CAM and the Python machine are on the same WiFi network.

---

## How it works

    ESP32-CAM → MJPEG stream (HTTP) → Python reads frames
    → YOLOv8n inference every 200ms → object counts + bounding boxes
    → Flask serves annotated video + JSON data → Browser dashboard (polling every 800ms)

Detection categories: person, vehicle, animal, bicycle, bag, phone, weapon,
umbrella, infra, furniture, object, others

Alert thresholds (triggers banner + audio beep):
- person:  5 or more detected
- weapon:  1 or more detected
- vehicle: 4 or more detected

---

## File structure

    server.py          Main Python app (Flask + YOLO + stream reader)
    yolov8n.pt          Downloaded automatically on first run
    eyecore_log.csv     Exported detection log (via dashboard Export button)
    eyecore_log.json    Exported detection log in JSON format
