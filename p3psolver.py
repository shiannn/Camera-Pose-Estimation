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

def p3psolver(points2D, points3D, cameraMatrix, distCoeffs=None):
    points2D = points2D[:3]
    points3D = points3D[:3]
    assert points2D.shape[0] == points3D.shape[0] and points3D.shape[0] == 3
    homo2D = np.concatenate([points2D, np.ones((points2D.shape[0],1))], axis=1)
    homo3D = np.concatenate([points3D, np.ones((points3D.shape[0],1))], axis=1)
    print(homo2D.shape)
    print(homo3D.shape)
    Kinv = np.linalg.inv(cameraMatrix)
    v = np.matmul(Kinv, homo2D.T).T
    ### find cosine angle Cab, Cac, Cbc
    unit_v = unit_vec(v)
    cosine2D = 1-spatial.distance.pdist(unit_v, metric='cosine')
    ### find Rab, Rac, Rbc
    pairdist3D = spatial.distance.pdist(points3D, metric='euclidean')
    coeff = calculateGterms(cosine2D, pairdist3D)
    ### solve polynomial
    root_x = np.roots(coeff)
    print(root_x)
    exit(0)

if __name__ == '__main__':
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    points2D = np.load('points2D.npy')
    points3D = np.load('points3D.npy')
    p3psolver(points2D, points3D, cameraMatrix)