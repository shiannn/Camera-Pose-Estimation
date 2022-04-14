from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time

from p3psolver import p3psolver

def average_desc(train_df, points3D_df):
    def average(x):
        return list(np.mean(x,axis=0))
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def load_data():
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    images_df = pd.read_pickle("data/images.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    return train_df, points3D_df, images_df, point_desc_df

def get2DImgDescriptor(idx, images_df, point_desc_df):
    # Load quaery image
    fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

    # Load query keypoints and descriptors
    points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
    kp_query = np.array(points["XY"].to_list())
    desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
    return rimg, kp_query, desc_query

def main():
    idx = 200
    ### load data
    train_df, points3D_df, images_df, point_desc_df = load_data()
    ### get 3D points & 3D descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list()) #### kp_model [num_3D_points, 3] [111519, 3]
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32) #### desc_model [num_3D_points, 128D] [111519, 128]
    print(kp_model.shape)
    ### get 2D query image & 2D descriptors
    rimg, kp_query, desc_query = get2DImgDescriptor(idx, images_df, point_desc_df)
    #### single image shape 1920* 1080
    #### kp_query [num_2D_points, 2] [4276, 2]
    #### desc_query [num_2D_points, 128D] [4276, 128]
    ### get 2D-3D correspondence
    points2D, points3D = get2D3Dcorrespondence((kp_query, desc_query), (kp_model, desc_model))
    #### points2D [num_correspondence, 2]
    #### points3D [num_correspondence, 3]
    np.save('points2D.npy', points2D)
    np.save('points3D.npy', points3D)
    ### p3p solver
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    p3psolver(points2D, points3D, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
    ### get ground_truth


def get2D3Dcorrespondence(query,model):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))
    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))
    
    return points2D, points3D

    #return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)

if __name__ == '__main__':
    main()

# Find correspondance and solve pnp
retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()
tvec = tvec.reshape(1,3)

# Get camera pose groudtruth 
ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
tvec_gt = ground_truth[["TX","TY","TZ"]].values

