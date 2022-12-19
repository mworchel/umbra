import numpy as np

def ray_plane_intersection(p0, n, l0, l):
    return np.dot((p0 - l0), n) / np.dot(l, n)

def to_opengl_matrix(A: np.ndarray):
    return np.ascontiguousarray(A.T).astype(np.float32)
