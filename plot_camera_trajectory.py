import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def plot_camera_object(Quaternion, Translation, focal_len_scaled=0.5, aspect_ratio=0.3, color=[1.0,0.,0.]):
    #print(Quaternion, Translation)
    rotation_Matrix = R.from_quat(Quaternion).as_matrix()
    points = np.array([
        [0, 0, 0],
        [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled],
        [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled],
        [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled],
        [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled]
    ])
    #points = np.matmul(rotation_Matrix, points.T).T + Translation
    points = np.matmul(rotation_Matrix.T,(points-Translation).T).T
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    #points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    #lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [color for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set, points[0]

def plot_point_cloud(points3D_df):
    pcd = o3d.geometry.PointCloud()
    xyz_list = points3D_df.iloc[:,1].to_numpy()
    xyz_array = np.stack((xyz_list),axis=0).astype(np.float64)
    rgb_list = points3D_df.iloc[:,2].to_numpy()
    rgb_array = np.stack((rgb_list),axis=0) / 255.
    print(xyz_array.dtype)
    print(rgb_array.dtype)
    
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    pcd.colors = o3d.utility.Vector3dVector(rgb_array)
    print(np.asarray(pcd.points))
    print(np.asarray(pcd.colors))
    return pcd

def parse_gt():
    def sort_func(x):
        x = x.str.replace('valid_img','')
        x = x.str.replace('train_img','')
        x = x.str.replace('.jpg','').astype(int)
        return x

    images_df = pd.read_pickle("data/images.pkl")
    valid_df = images_df[images_df["NAME"].str.contains("valid")]
    sorted_valid_df = valid_df.sort_values(by='NAME',key=sort_func)
    print(sorted_valid_df)

    translations = sorted_valid_df[['TX','TY','TZ']].to_numpy()
    quaternions = sorted_valid_df[["QX","QY","QZ","QW"]].to_numpy()
    print(translations.shape)
    print(quaternions.shape)
    np.save('data/translations.npy', translations)
    np.save('data/quaternions.npy', quaternions)

def main():
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = plot_point_cloud(points3D_df)
    o3d.io.write_point_cloud("data/gate.pcd", pcd)
    pcd = o3d.io.read_point_cloud("data/gate.pcd")
    print(np.asarray(pcd.points))
    print(np.asarray(pcd.colors))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    ### add points lines
    #o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    ### parse ground truth images extrinsic
    parse_gt()
    regists = [
        ('data/translations.npy', 'data/quaternions.npy', [1.,0.0,0.], [0.,1.0,0.0]),
        ('data/p3p_Trans.npy', 'data/p3p_Quats.npy', [0.,0.,1.0], [1.0,1.0,0.0])
    ]
    for trans_name, quats_name, color, traj_color in regists:
        translations = np.load(trans_name)
        quaternions = np.load(quats_name)
        
        trajectory_points = []
        for quaternion, translation in zip(quaternions[::1], translations[::1]):
            #translation = row[['TX','TY','TZ']].to_numpy()
            #quaternion = row[["QX","QY","QZ","QW"]].to_numpy()
            line_set, trajectory_point = plot_camera_object(quaternion, translation, color=color)
            vis.add_geometry(line_set)

            trajectory_points.append(trajectory_point)
        trajectory_points = np.stack(trajectory_points, axis=0)
        print(trajectory_points)
        trajectory_lines = np.stack([np.arange(trajectory_points.shape[0]-1), np.arange(1, trajectory_points.shape[0])], axis=1)

        camera_trajectory = o3d.geometry.LineSet()    
        camera_trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
        camera_trajectory.lines = o3d.utility.Vector2iVector(trajectory_lines)

        trajectory_colors = [traj_color for _ in range(trajectory_lines.shape[0])]
        camera_trajectory.colors = o3d.utility.Vector3dVector(trajectory_colors)
        vis.add_geometry(camera_trajectory)
    vis.run()

if __name__ == '__main__':
    main()
