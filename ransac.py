import numpy as np
from p3psolver import p3psolver
from scipy.spatial.transform import Rotation as R

def RansacSolveP3P(points2D, points3D, cameraMatrix, distCoeffs, n_iter=100, thres=1):
    def getBestFromMultipleSolution(FinalTs, Rotation_Matrixs):
        best_rotation_Matrix = None
        best_trans = None
        best_error = np.inf
        for trans, rotation_Matrix in zip(FinalTs, Rotation_Matrixs):
            outer = np.matmul(rotation_Matrix, points3D.T).T + trans
            onimg = np.matmul(cameraMatrix, outer.T).T
            onimg = onimg / onimg[:,2:]
            error = abs(onimg[:,:2] - points2D).mean()
            #print(error.mean())
            if best_rotation_Matrix is None or error < best_error:
                best_error = error
                best_rotation_Matrix = rotation_Matrix
                best_trans = trans
        best_quaternion = R.from_matrix(best_rotation_Matrix).as_quat()
        return best_trans, best_quaternion
    def project3Dto2D(rotation_Quat, translation, points3D, cameraMatrix):
        ### check result
        rotation_Matrix = R.from_quat(rotation_Quat).as_matrix()
        camera_coordinate = np.matmul(rotation_Matrix, points3D.T).T + translation
        img_nonhomogeneous = np.matmul(cameraMatrix, camera_coordinate.T).T
        img_homogeneous = img_nonhomogeneous / img_nonhomogeneous[:,2:]
        return img_homogeneous
    
    max_inliers = -1
    ret_T = None
    ret_Quat = None
    for it in range(n_iter):
        ### random select 3 2D-3D points correspondence
        select_correspondences = np.random.choice(points2D.shape[0], 3, replace=False)
        not_select = np.full(points2D.shape[0], True, dtype=bool)
        not_select[select_correspondences] = False
        ### p3p solver
        FinalTs, Rotation_Matrixs = p3psolver(
            points2D[select_correspondences], 
            points3D[select_correspondences], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs
        )
        ### skip this ransac iteration if no solution found from the 3 2D-3D correspondence
        if len(FinalTs) == 0 or len(Rotation_Matrixs) == 0:
            continue
        ### p3psolver return up to 4 solutions, select the best
        selected_T, selected_Quat = getBestFromMultipleSolution(FinalTs, Rotation_Matrixs)
        assert points3D[select_correspondences].shape[0] + points3D[not_select].shape[0] == points3D.shape[0]

        ### project 3D to 2D
        coordinates_2D = project3Dto2D(selected_Quat, selected_T, points3D[not_select], cameraMatrix=cameraMatrix)
        ### select model with most inliers
        num_inliers = (abs(coordinates_2D[:,:2] - points2D[not_select]).sum(axis=1)<thres).sum()
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            ret_T = selected_T
            ret_Quat = selected_Quat
        #print(max_inliers)
    return ret_Quat, ret_T, max_inliers