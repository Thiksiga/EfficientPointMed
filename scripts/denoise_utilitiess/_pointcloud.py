import numpy as np
import open3d as o3d
from ._basic_pointcloud import BasicPointCloud


def curvature_preserving_sampling(point_cloud: BasicPointCloud, k_neighbors=20, fraction=0.5):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
    # o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    curvatures = []
    kdtree = o3d.geometry.KDTreeFlann(o3d_cloud)
    for i in range(len(point_cloud.points)):
        _, idx, _ = kdtree.search_knn_vector_3d(o3d_cloud.points[i], k_neighbors)
        neighbors = np.asarray(o3d_cloud.points)[idx]
        # print(f"idx of {i}",idx)
        cov = np.cov(neighbors, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        curvature = eigenvalues[0] / sum(eigenvalues)
        curvatures.append(curvature)
    sorted_indices = np.argsort(curvatures)[::-1]
    # print(f"sorted indices of curvatures: ", len(sorted_indices))
    # print(f"sorted indices of curvatures: ", sorted_indices)
    # print(f"curvatures: {curvatures[sorted_indices]}")
    num_samples = int(len(point_cloud.points) * fraction)
    selected_indices = sorted_indices[:num_samples]
    return BasicPointCloud(points=point_cloud.points[selected_indices],
                        colors=point_cloud.colors[selected_indices],
                        normals=point_cloud.normals[selected_indices]), selected_indices


def curvature_preserving_sampling_2(point_cloud: BasicPointCloud, k_neighbors=20, fraction=0.2):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
    # o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    curvatures = []
    kdtree = o3d.geometry.KDTreeFlann(o3d_cloud)
    for i in range(len(point_cloud.points)):
        _, idx, _ = kdtree.search_knn_vector_3d(o3d_cloud.points[i], k_neighbors)
        neighbors = np.asarray(o3d_cloud.points)[idx]
        cov = np.cov(neighbors, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        curvature = eigenvalues[0] / sum(eigenvalues)
        curvatures.append(curvature)
    sorted_indices = np.argsort(curvatures)[::-1]
    num_samples = int(len(point_cloud.points) * fraction)
    selected_indices = sorted_indices[:num_samples]
    return BasicPointCloud(points=point_cloud.points[selected_indices], labels=point_cloud.labels[selected_indices], gradients=point_cloud.gradients[selected_indices], colors=point_cloud.colors[selected_indices], normals=point_cloud.normals[selected_indices]), selected_indices
