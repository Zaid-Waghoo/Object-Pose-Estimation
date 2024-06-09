import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from cv2 import aruco

def capture_realsense_images():
    """
    Capture color and depth images using an Intel RealSense camera.

    Returns:
        tuple: A tuple containing:
            - color_image (np.ndarray): The captured color image.
            - depth_image (np.ndarray): The captured depth image.
            - intrinsics (rs.intrinsics): The camera intrinsics.
            - intrinsics_matrix (np.ndarray): The camera intrinsics matrix.
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            intrinsics.model = rs.distortion.inverse_brown_conrady

            intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                          [0, intrinsics.fy, intrinsics.ppy],
                                          [0, 0, 1]])

            return color_image, depth_image, intrinsics, intrinsics_matrix
        
    finally:
        pipeline.stop()

def pose_detector(color_image, depth_image, intrinsics, centroids):
    """
    Detect the pose of objects in the scene using ArUco markers.

    Args:
        color_image (np.ndarray): The color image.
        depth_image (np.ndarray): The depth image.
        intrinsics (rs.intrinsics): The camera intrinsics.
        centroids (dict): Dictionary of object centroids.

    Returns:
        dict: Differences in pose for each object.
    """
    marker_size = 0.1
    offset = np.array([0, 0.2032, 0])
    
    centroids = centroids['data'][0]
    print(centroids)
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            intrinsics.model = rs.distortion.inverse_brown_conrady

            corners, aruco_centroid, rvecs, tvecs, mtx, dist = detect_markers(color_image, intrinsics, marker_size)
            if not corners:
                print("No ArUco markers detected.")
                continue

            depth_aruco_marker = get_median_depth(pipeline, align, profile, aruco_centroid[0])
            print(f"Median Depth: {depth_aruco_marker}")

            # Define rotation matrices
            rotation_90_deg_cw_z = np.array([-np.pi / 2, 0, 0])
            rotation_90_deg_cw_x = np.array([0, 0, -np.pi / 2])

            # Calculate rotation matrices
            camera_to_aruco_rotation = cv2.Rodrigues(rvecs[0])[0]
            rotation_matrix_90_deg_cw_z = cv2.Rodrigues(rotation_90_deg_cw_z)[0]
            rotation_matrix_90_deg_cw_x = cv2.Rodrigues(rotation_90_deg_cw_x)[0]
            aruco_to_base_rotation = rotation_matrix_90_deg_cw_z @ rotation_matrix_90_deg_cw_x

            # Calculate camera to base rotation
            camera_to_base = camera_to_aruco_rotation @ aruco_to_base_rotation
            new_rvecs = cv2.Rodrigues(camera_to_base)[0]

            # Calculate translation vectors
            translation_along_y_world = offset.reshape((3, 1))
            translation_along_y_camera = camera_to_base @ translation_along_y_world

            world_coords = pixel_to_3dpoint(aruco_centroid[0], depth_aruco_marker, intrinsics)
            new_tvecs = world_coords.reshape((3, 1)) + translation_along_y_camera

            # Retrieve object depths and 3D points
            object_depths = {
                name: get_median_depth(pipeline, align, profile, centroid) for name, centroid in centroids.items()
            }
            object_3d_points = {
                name: pixel_to_3dpoint(centroids[name], depth, intrinsics)
                for name, depth in object_depths.items()
            }

            if len(object_3d_points) < 1:
                print("Not enough centroids detected.")
                return

            # Reshape points for transformations
            world_coords_object = {name: point.reshape((3, 1)) for name, point in object_3d_points.items()}

            rotation_matrix = cv2.Rodrigues(new_rvecs)[0]
            transformed_tvecs = rotation_matrix @ new_tvecs
            transformed_world_coords_object = {
                name: rotation_matrix @ coord for name, coord in world_coords_object.items()
            }

            # Calculate differences
            differences = {
                name: (transformed - transformed_tvecs)[[2, 1, 0]]
                for name, transformed in transformed_world_coords_object.items()
            }

            # Negate the Y axis in the differences
            for difference in differences.values():
                difference[1] = -difference[1]

            for name, difference in differences.items():
                print(f"Difference for {name}: {difference}")

            # Draw axes on the image
            imaxis = cv2.drawFrameAxes(color_image.copy(), mtx, dist, new_rvecs, world_coords, 0.1)
            imaxis = cv2.drawFrameAxes(imaxis, mtx, dist, new_rvecs, new_tvecs, 0.1)

            for coord in world_coords_object.values():
                imaxis = cv2.drawFrameAxes(imaxis, mtx, dist, new_rvecs, coord, 0.1)

            plt.figure()
            plt.imshow(imaxis)
            plt.grid()
            plt.show()

            # Save the difference values
            np.save("differences.npy", differences)

            return differences

    finally:
        pipeline.stop()

def get_median_depth(pipeline, align, profile, centroid, num_frames=30):
    """
    Get the median depth value at a specific pixel location over a number of frames.

    Args:
        pipeline (rs.pipeline): The RealSense pipeline.
        align (rs.align): The alignment object to align depth frames to color frames.
        profile (rs.pipeline_profile): The pipeline profile.
        centroid (tuple): The (u, v) pixel coordinates.
        num_frames (int): The number of frames to capture for median depth calculation.

    Returns:
        float: The median depth value at the specified pixel location.
    """
    depth_values = []

    for _ in range(num_frames):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        u, v = centroid
        depth = aligned_depth_frame.get_distance(int(u), int(v))
        depth_values.append(depth)

    median_depth = np.median(depth_values)

    return median_depth

def pixel_to_3dpoint(centroid, depth, intrinsics):
    """
    Convert 2D pixel coordinates to 3D world coordinates.

    Args:
        centroid (tuple): The (u, v) pixel coordinates.
        depth (float): The depth value at the pixel location.
        intrinsics (rs.intrinsics): The camera intrinsics.

    Returns:
        np.ndarray: The 3D world coordinates.
    """
    u, v = centroid
    points = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)

    return np.array(points)

def detect_markers(frame, intrinsics, marker_size):
    """
    Detect ArUco markers in a frame and estimate their pose.

    Args:
        frame (np.ndarray): The image frame.
        intrinsics (rs.intrinsics): The camera intrinsics.
        marker_size (float): The size of the ArUco marker.

    Returns:
        tuple: A tuple containing:
            - corners (list): The detected marker corners.
            - centroids (list): The centroids of the detected markers.
            - rvecs (np.ndarray): The rotation vectors of the detected markers.
            - tvecs (np.ndarray): The translation vectors of the detected markers.
            - mtx (np.ndarray): The camera matrix.
            - dist (np.ndarray): The distortion coefficients.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if corners:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        for corner in corners:
            cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)

    mtx = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]])
    dist = np.zeros((5, 1))
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

    centroids = []
    for corner in corners:
        centroid = np.mean(corner[0], axis=0)
        centroids.append(centroid)

    return corners, centroids, rvecs, tvecs, mtx, dist
