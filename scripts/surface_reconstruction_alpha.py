import numpy as np
import open3d as o3d

data = np.load(r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_20_kn5.npy")
print("Loaded shape:", data.shape, "| dtype:", data.dtype)
pts = data[data[:, 4] == 1, :3]
# xyz = data[:, :3]
print("pcd shape:", pts.shape, "| dtype:", pts.dtype)

# build Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

alpha = 7
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
print(mesh)

mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

rest = o3d.geometry.VoxelGrid.create_from_triangle_mesh(input=mesh, voxel_size=0.3)
o3d.visualization.draw_geometries([rest], mesh_show_back_face=True)
print(rest)

# # np.save(r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\rest_20_kn5.npy", rest)
# mask = np.load(r"D:\Mphil\Programming\EfficientPointMed\dataset\P1-20250324T100522Z-001\P1\gt_alpha.npy")
# print(mask.shape)



