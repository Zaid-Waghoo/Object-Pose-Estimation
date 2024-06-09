import base64
import json
import os
import sys
import numpy as np
import requests

# Add the parent directory to the system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from realsense_functions import capture_realsense_images, pose_detector

class GradioClient:
    """
    A client to interact with a Gradio server for sending image, depth, and intrinsic data.

    Attributes:
        server_url (str): The URL of the Gradio server.
    """

    def __init__(self, server_url: str):
        """
        Initialize the GradioClient with the server URL.

        Args:
            server_url (str): The URL of the Gradio server.
        """
        self.server_url = server_url

    def encode_array_to_base64(self, array: np.ndarray) -> str:
        """
        Encode a NumPy array to a Base64 string.

        Args:
            array (np.ndarray): The array to encode.

        Returns:
            str: The Base64 encoded string.
        """
        array_bytes = array.tobytes()
        array_base64 = base64.b64encode(array_bytes).decode("utf-8")
        return array_base64

    def prepare_array(self, array: np.ndarray) -> str:
        """
        Prepare a NumPy array for JSON serialization.

        Args:
            array (np.ndarray): The array to prepare.

        Returns:
            str: JSON string of the array's Base64 encoded data, shape, and dtype.
        """
        array_base64 = self.encode_array_to_base64(array)
        data = {"array": array_base64, "shape": array.shape, "dtype": str(array.dtype)}
        return json.dumps(data)

    def send_request(self, image_array: np.ndarray, depth_array: np.ndarray, depth_intrinsic: np.ndarray):
        """
        Send image, depth, and intrinsic data to the Gradio server and get the response.

        Args:
            image_array (np.ndarray): The image data array.
            depth_array (np.ndarray): The depth data array.
            depth_intrinsic (np.ndarray): The depth intrinsic matrix.

        Returns:
            dict: The JSON response from the server.

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        # Prepare the arrays as JSON strings
        image_data_json = self.prepare_array(image_array)
        depth_data_json = self.prepare_array(depth_array)
        depth_intrinsic_json = self.prepare_array(depth_intrinsic)

        # Create the payload
        payload = {"data": [image_data_json, depth_data_json, depth_intrinsic_json]}
        headers = {"Content-Type": "application/json"}

        # Send the request to the Gradio server
        response = requests.post(self.server_url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

# Usage example
if __name__ == "__main__":
    # Update this URL if your Gradio server runs on a different address
    server_url = "http://80.188.223.202:11103/api/predict/"

    # Capture images and intrinsic data using RealSense camera
    image_array, depth_array, intrinsics, intrinsic_matrix = capture_realsense_images()

    # Initialize the GradioClient
    client = GradioClient(server_url)

    # Send the request and print the response
    try:
        response = client.send_request(image_array, depth_array, intrinsic_matrix)
        print("Response:", response)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

    # Detect object poses using the response from the Gradio server
    object_poses = pose_detector(image_array, depth_array, intrinsics, response)

    print("Object poses:", object_poses)

    # Save the object poses to a .npy file
    np.save("object_poses.npy", object_poses)
