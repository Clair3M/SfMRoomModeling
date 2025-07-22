import numpy as np
import open3d as o3d
import os

# returns a sorted list of strings - the file paths of images
# preconditions - the folder path exists
# postconditions - the file paths do not have the folder_path
def get_images(folder_path):
    images = []
    folder_content = os.listdir(folder_path)
    for file in folder_content:
        file_ext = file.split('.')[-1]
        if file_ext not in ["png", "jpg", "jpeg"]:
            continue
        images.append(folder_path + file)
    images.sort()
    return images


def get_camera_intrinsics(file_path):
    K = np.array([np.array([2759.48, 0, 1520.69], dtype=np.float32), 
        np.array([0, 2764.16, 1006.81], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32)], dtype=np.float32)
    return K


def print_point_cloud(point_cloud, file_name):
    try:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(file_name, pc)
    except Exception as e:
        print(e)
        return False
    return True


def print_usage(error_code):
    errors = [
        "ERROR: No image folder specified.",
        "ERROR: Invalid folder filepath."
        ]
    print(errors[error_code])
    print("Usage: main.py <image folder filepath> <camera instrinics (optional)>")
