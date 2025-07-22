# Structure From Motion (SfM) Room Modeling

This project creates a 3d model of an environment from photos of that environment using an SfM pipeline. This project was inspired by and is an extension of a project assigned in an image processing and vision course (CIS*4720 at the University of Guelph).

## Running the Program
To run the program input the following command into the command line:
```
./main.py <image folder path> <camera intrinsics file>
```
**\<image folder path\>** Contains the images from which the model will be created.

**\<camera intrinsics file\>** Contains the intrinsic matrix for the camera with which the images were taken. This must be a text file with the following format:
```
fx 0 cx
0 fy cy
0 0 1
```

## Requirements

This program is known to be working on Python version 3.11.11

### Required external modules

- [Numpy version 2.2.6](https://numpy.org/) 
- [Open3d version 0.19.0](https://www.open3d.org/)
- [OpenCV version 4.12.0.88](https://opencv.org/)

## Future Plans

- Add code to determine the intrinsic matrix of the camera.
- Map textures from the images to the point cloud.
- Create a physical device that rotates to capture images of the environment in which it sits then create a model of that environment.
