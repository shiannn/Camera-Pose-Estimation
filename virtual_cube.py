import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def get_virtual_cube(cube_vertices):
    def getFaces(minx, miny, minz, maxx, maxy, maxz):
        xarray = np.linspace(minx, maxx, 200)
        yarray = np.linspace(miny, maxy, 200)
        zarray = np.linspace(minz, maxz, 200)
        xygrid_x, xygrid_y  = np.meshgrid(xarray, yarray, indexing='ij')
        xzgrid_x, xzgrid_z = np.meshgrid(xarray, zarray, indexing='ij')
        yzgrid_y, yzgrid_z = np.meshgrid(yarray, zarray, indexing='ij')

        xy_face1 = np.stack([xygrid_x, xygrid_y, minz* np.ones_like(xygrid_x)], axis=2)
        xy_face2 = np.stack([xygrid_x, xygrid_y, maxz* np.ones_like(xygrid_x)], axis=2)

        xz_face1 = np.stack([xzgrid_x, miny* np.ones_like(xzgrid_x), xzgrid_z], axis=2)
        xz_face2 = np.stack([xzgrid_x, maxy* np.ones_like(xzgrid_x), xzgrid_z], axis=2)

        yz_face1 = np.stack([minx* np.ones_like(yzgrid_y), yzgrid_y, yzgrid_z], axis=2)
        yz_face2 = np.stack([maxx* np.ones_like(yzgrid_y), yzgrid_y, yzgrid_z], axis=2)

        xyz_cube = np.stack([xy_face1, xy_face2, xz_face1, xz_face2, yz_face1, yz_face2], axis=0)

        ### calculate RGB color
        rgb_colors = []
        for color_idx in range(6):
            #color = np.random.rand(*xy_face1.shape)#* np.ones_like(xy_face1)
            color = np.ones_like(xy_face1)
            #print(color.shape)
            #color[:,:] = np.array([1,1,0])
            color[:,:] = np.array([color_idx%2,color_idx//2%2,color_idx//2//2%2])
            rgb_colors.append(color)
        rgb_colors = np.stack(rgb_colors, axis=0)
        #print(rgb_colors.shape)
        xyzrgb_cube = np.concatenate((xyz_cube, rgb_colors), axis=3)
        return xyzrgb_cube
    print(cube_vertices)
    minx, miny, minz = cube_vertices.min(axis=0)
    maxx, maxy, maxz = cube_vertices.max(axis=0)
    print(minx, miny, minz)
    print(maxx, maxy, maxz)
    xyzrgb_grid = getFaces(minx, miny, minz, maxx, maxy, maxz)
    """
    xarray = np.linspace(minx, maxx, 50)
    yarray = np.linspace(miny, maxy, 50)
    zarray = np.linspace(minz, maxz, 50)
    xgrid, ygrid, zgrid = np.meshgrid(xarray, yarray, zarray, indexing='ij')
    xyz_grid = np.stack([xgrid, ygrid, zgrid], axis=3)
    ### calculate RGB from XYZ
    rarray = np.linspace(0, 1, 50)
    garray = np.linspace(0, 1, 50)
    barray = np.linspace(0, 1, 50)
    rgrid, ggrid, bgrid = np.meshgrid(rarray, garray, barray, indexing='ij')
    rgb_grid = np.stack([rgrid, ggrid, bgrid], axis=3)
    
    xyzrgb_grid = np.concatenate((xyz_grid, rgb_grid), axis=3)
    """
    return xyzrgb_grid

def project3Dto2D(rotation_Quat, translation, points3D, cameraMatrix):
    rotation_Matrix = R.from_quat(rotation_Quat).as_matrix()
    camera_coordinate = np.matmul(rotation_Matrix, points3D.T).T + translation
    img_nonhomogeneous = np.matmul(cameraMatrix, camera_coordinate.T).T
    img_homogeneous = img_nonhomogeneous / img_nonhomogeneous[:,2:]
    return img_homogeneous

def main():
    ### get virtual cube
    #cube_vertices = np.load('data/cube_vertices.npy')
    #cube_transform_mat = np.load('data/cube_transform_mat.npy')
    cube_vertices = np.array([
        [ 0.69,0.          ,0.83      ],
        [ 1.69,0.          ,0.83      ],
        [ 0.69,-0.60181502 , 1.62863551],
        [ 1.69,-0.60181502 , 1.62863551],
        [ 0.69,0.79863551  ,1.43181502],
        [ 1.69,0.79863551  ,1.43181502],
        [ 0.69,0.19682049  ,2.23045053],
        [ 1.69,0.19682049  ,2.23045053]
    ])
    cube_vertices[:,2] -= 1
    xyzrgb_grid = get_virtual_cube(cube_vertices)
    ### camera intrinsic matrix
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    ### get images extrinsic
    images_df = pd.read_pickle("data/images.pkl")
    points3D = xyzrgb_grid.reshape(-1,6)[:,:3]
    RGB3D = xyzrgb_grid.reshape(-1,6)[:,3:]
    
    valid_df = images_df[images_df["NAME"].str.contains("valid")]
    print(valid_df)

    writer = cv2.VideoWriter(
        filename='virtual_cube_AR.mp4', apiPreference=0,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=15, frameSize=(1080,1920)
    )
    ### sort valid_df['NAME']
    def sort_func(x):
        return x.str.replace('valid_img','').str.replace('.jpg','').astype(int)
    sorted_valid_df = valid_df.sort_values(by='NAME',key=sort_func)
    print(sorted_valid_df)
    for idx, row in sorted_valid_df.iterrows():
        A = cv2.imread(os.path.join(os.path.join('data','frames'),row['NAME']))
        
        translation = row[['TX','TY','TZ']].to_numpy()
        quaternion = row[["QX","QY","QZ","QW"]].to_numpy()
        # print(translation)
        # print(quaternion)
        coord2D = project3Dto2D(quaternion, translation, points3D, cameraMatrix)
        ### concatenate rgb
        coord2DRGB = np.concatenate((coord2D[:,:2], RGB3D), axis=1)
        ### remove outsiders
        coord2DRGB = coord2DRGB[(coord2DRGB[:,1]<1920) & (coord2DRGB[:,0]<1080) & (coord2DRGB[:,1]>=0) & (coord2DRGB[:,0]>=0)]
        ### reassign RGB value
        #print(coord2DRGB.shape)
        new_A = A.copy()
        for r in [0]:
            for c in [0]:
                new_A[coord2DRGB[:,1].astype(int)+r,coord2DRGB[:,0].astype(int)+c] = 255*coord2DRGB[:,2:]
        #print(abs(new_A - A).mean())
        #if abs(new_A - A).mean() > 0:
        writer.write(new_A)
        #cv2.imwrite('video_sequence/{}'.format(row['NAME']), new_A)
        #exit(0)
        #if idx == 20:
        #    break
    writer.release() 
if __name__ == '__main__':
    main()