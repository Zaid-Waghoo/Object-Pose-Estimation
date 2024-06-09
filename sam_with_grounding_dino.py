from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from utils import BoundingBox, DetectionResult, calculate_centroid, plot_detections, load_image, get_boxes, refine_masks

def detect(
    image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.

    Args:
        image (Image.Image): The input image.
        labels (List[str]): The list of labels to detect.
        threshold (float): The confidence threshold for detection.
        detector_id (Optional[str]): The model ID for the detector.

    Returns:
        List[Dict[str, Any]]: The detection results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)

    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image and a set of bounding boxes.

    Args:
        image (Image.Image): The input image.
        detection_results (List[Dict[str, Any]]): The detection results with bounding boxes.
        polygon_refinement (bool): Whether to refine the masks using polygon refinement.
        segmenter_id (Optional[str]): The model ID for the segmenter.

    Returns:
        List[DetectionResult]: The detection results with masks.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def detect_and_segment(
    image: Union[str, np.ndarray, Image.Image],
    labels: List[str],
    threshold: float = 1.0,
    detector_id: str = "IDEA-Research/grounding-dino-base",
    segmenter_id: str = "facebook/sam-vit-huge",
    polygon_refinement: bool = False,
    save_name: str = "experimental/images_for_segmentation/Segmented_image.png",
) -> Tuple[np.ndarray, List[DetectionResult]]:
    """
    Detect and segment objects in an image.

    Args:
        image (Union[str, np.ndarray, Image.Image]): The input image or path to the image.
        labels (List[str]): The list of labels to detect.
        threshold (float): The confidence threshold for detection.
        detector_id (str): The model ID for the detector.
        segmenter_id (str): The model ID for the segmenter.
        polygon_refinement (bool): Whether to refine the masks using polygon refinement.
        save_name (str): The path to save the segmented image.

    Returns:
        Tuple[np.ndarray, List[DetectionResult]]: The segmented image array and detection results.
    """
    image = load_image(image)
    
    detections = detect(image, labels, threshold, detector_id)

    detections = segment(image, detections, polygon_refinement, segmenter_id)

    image_array = np.array(image)
    plot_detections(image_array, detections, save_name)

    return detections

if __name__ == "__main__":
    # Directly pass the arguments to the function
    detect_and_segment(
        image="captured_images/images_for_segmentation/rgb_image.png",
        labels=["Robot", "Remote control", "Blue Memory Card", "White Charger", "Whiteboard Eraser", "Tray"],
        threshold=0.3,
        detector_id="IDEA-Research/grounding-dino-base",
        segmenter_id="facebook/sam-vit-huge",
        polygon_refinement=True,
        save_name="experimental/images_for_segmentation/Segmented_image.png",
    )
