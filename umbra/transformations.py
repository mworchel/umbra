import math
import numpy as np

def create_perspective_projection_matrix(fovy, aspect, near, far):
    tan_half_fovy = math.tan(0.5*math.radians(fovy))
    return np.array([[1/(aspect*tan_half_fovy),    0,                       0,                         0],
                     [  0,   1/tan_half_fovy,                      0,                         0],
                     [  0,        0, -(far+near)/(far-near),  -(2*far*near)/(far-near)],
                     [  0,        0,                     -1,                         0]], dtype=np.float32)

def create_translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=np.float32)

def create_lookat_matrix(eye, focus, up):
    z = eye - focus
    x = np.cross(up, z)
    if np.isclose(np.linalg.norm(x), 0.0):
        x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y = np.cross(z, x)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    R = np.stack([x, y, z], axis=0)
    R = np.hstack([R, np.array([[0], [0], [0]], dtype=np.float32)])
    R = np.vstack([R, np.array([[0, 0, 0, 1]], dtype=np.float32)])

    T = create_translation_matrix(-eye[0], -eye[1], -eye[2])

    return R @ T