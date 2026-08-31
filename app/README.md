# Client Application (`app/`)

This directory contains the codebase for the **Mobile/Dashcam Client Application**. 

## 🎯 Purpose
The client app is the edge component of our system. It is designed to be run on edge devices (like smartphones mounted on vehicle dashboards or dedicated dashcams). 

Its primary responsibilities are:
1. **Video Capture:** Continuously capturing video frames from the device's camera.
2. **GPS Logging:** Acquiring real-time GPS coordinates to geotag video frames.
3. **Edge Inference (Planned):** Running the exported ONNX models (from the `ml/` pipeline) directly on the device to detect road anomalies without relying on a constant internet connection.
4. **Data Synchronization:** Uploading detected anomaly events (type, severity, location, timestamp) to the backend server.

## 🚧 Status
*This component is currently under active development. Stay tuned for setup instructions and technology stack details.*
