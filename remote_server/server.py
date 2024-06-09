import base64
import json
from typing import Dict

import gradio as gr
import numpy as np

import sys
import os

# Add the parent directory to the system path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sam_with_grounding_dino import detect_and_segment

def decode_base64_to_array(base64_str: str, shape: tuple, dtype: str) -> np.ndarray:
    """
    Decode a Base64 string to a NumPy array.

    Args:
        base64_str (str): The Base64 encoded string.
        shape (tuple): The shape of the array.
        dtype (str): The data type of the array.

    Returns:
        np.ndarray: The decoded NumPy array.
    """
    array_bytes = base64.b64decode(base64_str)
    array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    return array

def process_images(image_data: str, depth_data: str, depth_intrinsic_data: str) -> Dict:
    """
    Process image, depth, and intrinsic data to detect and segment objects.

    Args:
        image_data (str): JSON string of the image array in Base64 format.
        depth_data (str): JSON string of the depth array in Base64 format.
        depth_intrinsic_data (str): JSON string of the depth intrinsic array in Base64 format.

    Returns:
        Dict: A dictionary of detected object labels and their 3D centroids.
    """
    # Decode the image array from JSON
    image_data_json = json.loads(image_data)
    image_array = decode_base64_to_array(
        image_data_json["array"], tuple(image_data_json["shape"]), image_data_json["dtype"]
    )

    # Decode depth array from JSON
    depth_data_json = json.loads(depth_data)
    depth_array = decode_base64_to_array(
        depth_data_json["array"], tuple(depth_data_json["shape"]), depth_data_json["dtype"]
    )

    # Decode depth intrinsic array from JSON
    depth_intrinsic_json = json.loads(depth_intrinsic_data)
    depth_intrinsic_array = decode_base64_to_array(
        depth_intrinsic_json["array"], tuple(depth_intrinsic_json["shape"]), depth_intrinsic_json["dtype"]
    )

    # Detect and segment objects in the image
    detection_results = detect_and_segment(
        image=image_array,
        labels=["Remote control", "Blue Memory Card", "Black Tray"],
        threshold=0.3,
        detector_id="IDEA-Research/grounding-dino-base",
        segmenter_id="facebook/sam-vit-huge",
        polygon_refinement=True,
        save_name="experimental/images_for_segmentation/Segmented_image.png",
    )
    
    # Extract centroids and labels from the detection results
    centroids = [detection.centroid for detection in detection_results]
    labels = [detection.label for detection in detection_results]

    # Make a dictionary of the centroids
    centroids = {label: centroid for label, centroid in zip(labels, centroids)}
    
    return centroids

# Create a Gradio interface for the process_images function
interface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Textbox(label="Image Array (Base64 JSON)"),
        gr.Textbox(label="Depth Array (Base64 JSON)"),
        gr.Textbox(label="Depth Intrinsic (Base64 JSON)"),
    ],
    outputs=gr.JSON(label="3D Pose"),
)

if __name__ == "__main__":
    # Launch the Gradio interface
    interface.launch(server_name="0.0.0.0", server_port=8081)
