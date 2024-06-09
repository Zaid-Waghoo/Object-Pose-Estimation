Object Detection and Segmentation with Grounding DINO and SAM

This project provides a pipeline for detecting and segmenting objects in images using Grounding DINO for detection and Segment Anything (SAM) for segmentation.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Server](#server)
  - [Client](#client)

## Requirements

- Python 3.8+
- Intel RealSense camera (for capturing images)
- CUDA (for GPU support, optional but recommended)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/object-detection-segmentation.git
    cd object-detection-segmentation
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Server

1. **Run the server:**

    ```sh
    python server.py
    ```

    The server will start on `http://0.0.0.0:8081`.

2. **Server Description:**
   
   The server processes images, performs object detection and segmentation, and returns the results. It uses the `detect_and_segment` function from the `sam_with_grounding_dino` module.

### Client

1. **Run the client:**

    ```sh
    python client.py
    ```

2. **Client Description:**
   
   The client captures images using an Intel RealSense camera and sends them to the server for processing. The client uses the `GradioClient` class to send requests to the server and receive the processed results. It also uses functions from the `realsense_functions` module to capture images and detect poses. 

.
