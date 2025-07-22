#!/usr/bin/env python3

import sys
import os
import time

from source.system_functions import get_images, print_usage, get_camera_intrinsics,\
    print_point_cloud
from source.struct_from_motion import create_pointcloud

def main():
    try: 
        if not os.path.exists(sys.argv[1]):
            raise ValueError
        image_folder = sys.argv[1]
        if image_folder[-1] != '/':
            image_folder += '/'
        image_files = get_images(image_folder)
    except ValueError: 
        print_usage(1)
    except:
        print_usage(0)
    # read command line arguments for image folder, camera intrinsics, etc.
    cam_intrinsics = get_camera_intrinsics(None)
    start_time = time.perf_counter()
    point_cloud = create_pointcloud(image_files, cam_intrinsics)
    end_time = time.perf_counter()
    print_point_cloud(point_cloud, "./ahhh.ply")
    print(f"Time to create point cloud: {(end_time-start_time):.2f}s for {len(point_cloud)} points")
    return 0

if __name__ == "__main__":
    main()
