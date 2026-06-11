import numpy as np


class BasicPointCloud:
    def __init__(self, points, colors=None, normals=None):
        self.points = points
        N = points.shape[0]
        # self.gradients = gradients
        # self.labels = labels
        self.colors = colors if colors is not None else np.zeros_like(points)
        self.normals = normals if normals is not None else np.zeros_like(points)
        # self.gradients = gradients if gradients is not None else np.zeros((N, 1), np.float32)
        # self.labels = labels if labels is not None else np.full((N,), -1, np.int32)
        # self.colors = colors if colors is not None else np.zeros((N, 3), points.dtype)
        # self.normals = normals if normals is not None else np.zeros((N, 3), points.dtype)
