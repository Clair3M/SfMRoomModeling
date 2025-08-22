import cv2 as cv
import numpy as np
from scipy import stats

# returns keypoint pixel loactions and sift descriptors for the keypoints
def get_keypoints(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None, None
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def get_matching_kps_flann(desc1, desc2):
    
    return None


# returns empty list on failure
def get_matching_kps_bf(desc1, desc2):
    bf = cv.BFMatcher()
    # returns the two best keypoint matches in desc2 for each desc1 point
    matches = bf.knnMatch(desc1, desc2, k=2)
    # gets all matching points that are sufficiently unique
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    return good_matches


# returns two ordered lists of img keypoints (or descriptors) corresponding to matches
def get_matching_kps(matches, img1_points, img2_points):
    if 0 in [len(matches), len(img1_points), len(img2_points)] \
            or len(matches) > len(img1_points) \
            or len(matches) > len(img2_points):
        print("Error: invalid parameters")
        return None, None
    img1_kps = []
    img2_kps = []
    try:
        for match in matches:
            match = match[0]
            img1_kps.append(img1_points[match.queryIdx])
            img2_kps.append(img2_points[match.trainIdx])
    except Exception as e: 
        print(str(e))
        return None, None
    return img1_kps, img2_kps


# returns 3x4 [Rt] numpy projection matrix for given image
def get_projection_matrix(K, img1_kps=None, img2_kps=None, identity=False):
    if identity:
        rotation = np.identity(3)
        translation = np.zeros((3, 1))
    elif (img1_kps.shape[1] == 2):
        E, E_mask = cv.findEssentialMat(img1_kps, img2_kps, K,\
                                        method=cv.RANSAC, prob=0.99, threshold=1.0)
        num_inliers, rotation, translation, mask = cv.recoverPose(E, img1_kps,\
                                                                  img2_kps, mask=E_mask)
    elif (img1_kps.shape[1] == 3):
        success, rotation_vector, translation, inlier = cv.solvePnPRansac(\
            img1_kps, img2_kps, K, distCoeffs=None, iterationsCount=100,\
            reprojectionError=8.0,confidence=0.99)
        rotation, dumby = cv.Rodrigues(rotation_vector)
    proj_mat = np.concatenate((rotation, translation), axis=1)
    proj_mat = np.matmul(K, proj_mat)
    return proj_mat


def convert_3d_norm_to_3d(points_4d):
    if points_4d is None or points_4d.shape[0] != 4:
        return None
    points_3d = []
    for i in range(len(points_4d[0])):
        new_point = np.array([points_4d[0][i], points_4d[1][i], \
                               points_4d[2][i]], dtype=np.float32)
        new_point = np.divide(new_point, points_4d[3][i])
        points_3d.append(new_point)
    return points_3d

def remove_outliers(points, max_z_score):
    points = np.array(points)
    normalized_points = np.linalg.norm(points, axis=1)
    z = np.abs(stats.zscore(normalized_points))
    outlier_indices = np.where(z > max_z_score)[0]
    refined_points = np.delete(points, outlier_indices, axis=0)
    refined_points = refined_points.tolist()
    return refined_points


def create_pointcloud(img_list, K):
    img1_points, img1_desc = get_keypoints(img_list[0])
    img2_points, img2_desc = get_keypoints(img_list[1])
    init_matches = get_matching_kps_bf(img1_desc, img2_desc)

    if init_matches == []:
        print("add error handling here :) (initial keypoint matches fail)")

    img1_matches, img2_matches = get_matching_kps(init_matches, img1_points,\
                                                  img2_points)
    img1_mat_desc, img2_mat_desc = get_matching_kps(init_matches, img1_desc,\
                                                    img2_desc)

    # check that points are valid and not none
    img1_kps_xy = np.array([kp.pt for kp in img1_matches], dtype=np.float32)
    img2_kps_xy = np.array([kp.pt for kp in img2_matches], dtype=np.float32)

    proj_matrix1 = get_projection_matrix(K, identity=True)
    proj_matrix2 = get_projection_matrix(K, img1_kps_xy, img2_kps_xy)

    img1_kps_xy = img1_kps_xy.reshape(-1, 1, 2)
    img2_kps_xy = img2_kps_xy.reshape(-1, 1, 2)
    points_4d = cv.triangulatePoints(proj_matrix1, proj_matrix2, img1_kps_xy,\
                                     img2_kps_xy)
    point_cloud = convert_3d_norm_to_3d(points_4d)
    #point_cloud = remove_outliers(points_3d, 2)

    match_trainIdx = [match[0].trainIdx for match in init_matches]
    match_trainIdx.sort(reverse=True)
    prev_kps = np.array(img2_points)
    prev_descs = img2_desc
    for match_id in match_trainIdx:
        prev_kps = np.delete(prev_kps, match_id, 0)
        prev_descs = np.delete(prev_descs, match_id, 0)
    prev_proj_mat = proj_matrix2
    point_cloud_descs = np.array(img2_mat_desc, dtype=np.float32)

    for img in img_list[2:5]:
        new_img_points, new_img_desc = get_keypoints(img)
        matches_2d_3d = get_matching_kps_bf(point_cloud_descs, new_img_desc)
        kp_matches_3d, kp_matches_2d = get_matching_kps(matches_2d_3d,\
                                                            point_cloud,\
                                                            new_img_points)
        kp_matches_3d = np.array(kp_matches_3d, dtype=np.float32)
        kp_matches_2d = np.array([kp.pt for kp in kp_matches_2d], dtype=np.float32)
        proj_mat = get_projection_matrix(K, kp_matches_3d, kp_matches_2d)

        matches_2d_2d = get_matching_kps_bf(prev_descs, new_img_desc)
        old_matches, new_matches = get_matching_kps(matches_2d_2d, prev_kps,\
                                                    new_img_points)
        old_match_descs, new_match_descs = get_matching_kps(matches_2d_2d, prev_descs,\
                                                    new_img_desc)
        old_kps_xy = np.array([kp.pt for kp in old_matches], dtype=np.float32)
        new_kps_xy = np.array([kp.pt for kp in new_matches], dtype=np.float32)
        old_kps_xy = old_kps_xy.reshape(-1, 1, 2)
        new_kps_xy = new_kps_xy.reshape(-1, 1, 2)
        
        new_4d_points = cv.triangulatePoints(prev_proj_mat, proj_mat, old_kps_xy, new_kps_xy)
        new_3d_points = convert_3d_norm_to_3d(new_4d_points)

        new_3d_points = remove_outliers(new_3d_points, 2)

        point_cloud += new_3d_points
        match_trainIdx = [match[0].trainIdx for match in matches_2d_2d]
        match_trainIdx.sort(reverse=True)
        prev_kps = np.array(new_img_points)
        prev_descs = new_img_desc
        for match_id in match_trainIdx:
            prev_kps = np.delete(prev_kps, match_id, 0)
            prev_descs = np.delete(prev_descs, match_id, 0)
        prev_proj_mat = proj_mat
        point_cloud_descs = np.append(point_cloud_descs, new_match_descs, axis=0)
    return point_cloud
