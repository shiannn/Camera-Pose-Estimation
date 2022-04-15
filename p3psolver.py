import numpy as np
from scipy import spatial

def unit_vec(vec):
    assert len(vec.shape) == 2
    vec_norm = np.linalg.norm(vec, ord=2, axis=1, keepdims=True)
    unit_vector = vec / vec_norm
    assert (np.linalg.norm(unit_vector, ord=2, axis=1) - 1).mean() < 1e-3
    return unit_vector

def calculateGterms(cosine2D, pairdist3D):
    Cab, Cac, Cbc = cosine2D
    Rab, Rac, Rbc = pairdist3D
    K1 = (Rbc/Rac)**2
    K2 = (Rbc/Rab)**2
    #print(K1, K2)
    G0 = (K1*K2+K1-K2)**2 - 4*K1**2*K2*Cac**2
    #print(G0)
    G1 = 4*(K1*K2+K1-K2)*K2*(1-K1)*Cab + 4*K1*((K1*K2-K1+K2)*Cac*Cbc+2*K1*K2*Cab*Cac**2)
    #print(G1)
    G2 = (2*K2*(1-K1)*Cab)**2+2*(K1*K2-K1-K2)*(K1*K2+K1-K2) + 4*K1*((K1-K2)*Cbc**2+K1*(1-K2)*Cac**2-2*(1+K1)*K2*Cab*Cac*Cbc)
    #print(G2)
    G3 = 4*(K1*K2-K1-K2)*K2*(1-K1)*Cab + 4*K1*Cbc*((K1*K2-K1+K2)*Cac+2*K2*Cab*Cbc)
    #print(G3)
    G4 = (K1*K2-K1-K2)**2 - 4*K1*K2*Cbc**2
    #print(G4)
    coeff = np.array([G4,G3,G2,G1,G0])
    return coeff

def solve_a(xs, pairdist3D, cosine2D):
    Cab, Cac, Cbc = cosine2D
    Rab, Rac, Rbc = pairdist3D
    ret_xas = []
    for x in xs:
        coeff = np.array([1+x**2-2*x*Cab, 0.0, -Rab**2])
        root_as = np.roots(coeff)
        for a in root_as:
            ret_xas.append((x, a))
    return ret_xas

def solve_y(xas, pairdist3D, cosine2D):
    Cab, Cac, Cbc = cosine2D
    Rab, Rac, Rbc = pairdist3D
    ret_xyas = []
    for x,a in xas:
        #print('a', a, 'x',x)
        coeff = np.array([a**2, -2*a**2*Cac, a**2-Rac**2])
        root_ys = np.roots(coeff)
        for y in root_ys:
            ret_xyas.append((x,y,a))
    return ret_xyas

def findTpoint(solution_xya, points3D, unit_v):
    def transform2originxy(points3D):
        x1,x2,x3 = points3D
        d12 = np.linalg.norm(x1-x2,ord=2)
        d23 = np.linalg.norm(x2-x3,ord=2)
        d13 = np.linalg.norm(x1-x3,ord=2)
        f1 = np.array([0.,0.,0.])
        f2 = np.array([d12,0.,0.])
        cos = 1-spatial.distance.cosine(x2-x1,x3-x1)
        sin = np.sin(np.arccos(cos))
        f3 = np.array([d13*cos,d13*sin,0.])
        ### check
        df12 = np.linalg.norm(f1-f2,ord=2)
        df23 = np.linalg.norm(f2-f3,ord=2)
        df13 = np.linalg.norm(f1-f3,ord=2)
        assert abs(df12 - d12) < 1e-3 and abs(df23 - d23) < 1e-3 and abs(df13 - d13) < 1e-3
        return np.array([f1,f2,f3])

    def getRigidTransform(A, B):
        ### return RigidTransform from A to B
        assert A.shape[1] == 3 and B.shape[1] == 3
        centroidA = A.mean(axis=0, keepdims=True)
        centroidB = B.mean(axis=0, keepdims=True)
        ### H is an covariance matrix 3xN matmul Nx3
        H = np.matmul((A - centroidA).T, (B-centroidB))
        U,S,Vh = np.linalg.svd(H)
        R = np.matmul(Vh.T,U.T)
        if np.linalg.det(R) < 0:
            #print("det(R) < R, reflection detected!, correcting for it ...")
            Vh[2,:] *= -1
            #R = Vt.T @ U.T
            R = np.matmul(Vh.T,U.T)
        T = centroidB.T - np.matmul(R, centroidA.T)
        ### check
        assert (B-(np.matmul(R, A.T) + T).T).mean() < 1e-5
        return R,T
    
    def trilaterationXY(r1,r2,r3,U,Vx,Vy):
        x = (r1**2-r2**2+U**2)/(2*U)
        y = (r1**2-r3**2+Vx**2+Vy**2-2*Vx*x)/(2*Vy)
        z = np.sqrt(r1**2-x**2-y**2)
        return [np.array([x,y,z]),np.array([x,y,-z])]

    def solveTrilateration(radius, points3D, unit_v):
        x1,x2,x3 = points3D
        a,b,c = radius
        v1,v2,v3 = unit_v
        points3D_XYplane = transform2originxy(points3D)
        Rotate,Translation = getRigidTransform(points3D, points3D_XYplane)
        ### Translation 3x1
        #print(points3D_XYplane)
        TpointXYs = trilaterationXY(
            a,b,c,
            points3D_XYplane[1,0],points3D_XYplane[2,0],points3D_XYplane[2,1]
        )
        #print(TpointXY)
        ### check
        for TpointXY in TpointXYs:
            d1 = np.linalg.norm(TpointXY-points3D_XYplane[0],ord=2)
            d2 = np.linalg.norm(TpointXY-points3D_XYplane[1],ord=2)
            d3 = np.linalg.norm(TpointXY-points3D_XYplane[2],ord=2)
            #print('abs(d1-a), abs(d2-b), abs(d3-c)', abs(d1-a), abs(d2-b), abs(d3-c))
            #assert abs(d1-a)<1e-3 and abs(d2-b)<1e-3 and abs(d3-c)<1e-3
        Tpoint_Errors = []
        ### inverse TpointXY back to 3d space
        for TpointXY in TpointXYs:
            Tpoint = np.matmul(np.linalg.inv(Rotate), (TpointXY.reshape(-1,1) - Translation))
            Tpoint = Tpoint.squeeze()
            ### check in 3D
            d31 = np.linalg.norm(Tpoint-x1,ord=2)
            d32 = np.linalg.norm(Tpoint-x2,ord=2)
            d33 = np.linalg.norm(Tpoint-x3,ord=2)
            #assert abs(d31-a)<1e-3 and abs(d32-b)<1e-3 and abs(d33-c)<1e-3
            error = (abs(d31-a)+abs(d32-b)+abs(d33-c))/3.
            Tpoint_Errors.append((Tpoint,error))
        return Tpoint_Errors
    Tpoints = []
    for x,y,a in solution_xya:
        #if np.isreal(x) and np.isreal(y) and np.isreal(a):
        b = x*a
        c = y*a
        Tpoint_Errors = solveTrilateration((a,b,c), points3D, unit_v)
        for Tpoint, error in Tpoint_Errors:
            if error < 1e-3:
                Tpoints.append(Tpoint)
    return Tpoints

def p3psolver(points2D, points3D, cameraMatrix, distCoeffs=None):
    assert len(points2D.shape) == 2 and points2D.shape[0] == 3
    assert len(points3D.shape) == 2 and points3D.shape[0] == 3
    homo2D = np.concatenate([points2D, np.ones((points2D.shape[0],1))], axis=1)
    homo3D = np.concatenate([points3D, np.ones((points3D.shape[0],1))], axis=1)
    #print(homo2D.shape)
    #print(homo3D.shape)
    Kinv = np.linalg.inv(cameraMatrix)
    v = np.matmul(Kinv, homo2D.T).T
    ### find cosine angle Cab, Cac, Cbc
    unit_v = unit_vec(v)
    cosine2D = 1-spatial.distance.pdist(unit_v, metric='cosine')
    ### find Rab, Rac, Rbc
    pairdist3D = spatial.distance.pdist(points3D, metric='euclidean')
    ### solve Gterms polynomial
    coeff = calculateGterms(cosine2D, pairdist3D)
    root_x = np.roots(coeff)
    ### solve a
    root_as = solve_a(root_x, pairdist3D, cosine2D)
    ### solve y
    solution_xya = solve_y(root_as, pairdist3D, cosine2D)
    Tpoints = findTpoint(solution_xya, points3D, unit_v)
    ### calculated lambda with all possible Tpoints
    lambdas = []
    for Tpoint in Tpoints:
        right = points3D-np.expand_dims(Tpoint, axis=0)
        right_norm = np.linalg.norm(right, ord=2, axis=1)
        #print(right)
        #print(right_norm)
        left_norm = np.linalg.norm(v, ord=2, axis=1)
        #print(left_norm)
        lamb = np.divide(right_norm, left_norm)
        #print(lamb)
        lambdas.append(lamb)
    #print(lambdas)
    ### calculate rotation matrix R
    Rotation_Matrixs = []
    FinalTs = []
    #print(Tpoints)
    for lamb, Tpoint in zip(lambdas, Tpoints):
        left_matrix = lamb.reshape(-1,1)*v
        right_matrix = points3D-np.expand_dims(Tpoint, axis=0)
        #print(left_matrix)
        #print(right_matrix)
        Rotation_Matrix = np.matmul(left_matrix.T,np.linalg.inv(right_matrix.T))
        ### check Rotation Matrix
        if not abs(np.linalg.det(Rotation_Matrix) - 1) < 1e-3:
            continue
        orthonormal = np.matmul(Rotation_Matrix.T, Rotation_Matrix)
        if not abs(orthonormal - np.eye(orthonormal.shape[0])).mean() < 1e-3:
            continue
        Rotation_Matrixs.append(Rotation_Matrix)
        ### transformation due to the notation difference on slide & hw
        TransTpoint = -np.matmul(Rotation_Matrix, Tpoint)
        FinalTs.append(TransTpoint)
    #print(FinalTs)
    #print(Rotation_Matrixs)
    return FinalTs, Rotation_Matrixs

if __name__ == '__main__':
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    points2D = np.load('points2D.npy')
    points3D = np.load('points3D.npy')
    FinalTs, Rotation_Matrixs = p3psolver(points2D[:3], points3D[:3], cameraMatrix)
    for trans, rotation_Matrix in zip(FinalTs, Rotation_Matrixs):
        rot_object = spatial.transform.Rotation.from_matrix(rotation_Matrix)
        rotq = rot_object.as_quat()
        ### check result
        outer = np.matmul(rotation_Matrix, points3D.T).T + trans
        onimg = np.matmul(cameraMatrix, outer.T).T
        onimg = onimg / onimg[:,2:]
        print(onimg)
        print(points2D)
        #np.matmul(rotation_Matrix, points3D.T) + trans