import numpy as np
import open3d as o3d

# --- config ---
# file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_3.npy"
point_size = 1.0   # try 1.5–3.0 if points look too thin
voxel_threshold_for_grid = 0  # if the array is a 3D voxel grid, treat > threshold as occupied

def visualize_label1(pc_with_label: np.ndarray,
                     point_size: float = 3.0,
                     window_name: str = "Masked Region (label == 1)",
                     recenter: bool = True):
    """
    Visualize only the masked region (label == 1) from a point cloud shaped (N,5): [x,y,z,grad,label].
    Keeps your main script unchanged.
    """
    import numpy as np
    import open3d as o3d

    if pc_with_label.ndim != 2 or pc_with_label.shape[1] < 5:
        raise ValueError(f"Expected (N,5) array [x,y,z,grad,label]; got {pc_with_label.shape}")

    labels = pc_with_label[:, 4]
    # Be robust to float labels (0.0/1.0) as well as ints
    if np.issubdtype(labels.dtype, np.integer):
        mask = (labels == 1)
    else:
        mask = (labels >= 0.5)

    pts = pc_with_label[mask, :3].astype(np.float64, copy=False)
    if pts.size == 0:
        print("No points with label == 1.")
        return

    if recenter:
        pts = pts - pts.mean(axis=0, keepdims=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([1.0, 1.0, 1.0])  # white for contrast

    # small axis for orientation
    extent = (pts.max(axis=0) - pts.min(axis=0))
    axis_size = float(np.max(extent) / 20.0) if np.all(np.isfinite(extent)) and np.max(extent) > 0 else 1.0
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=800, visible=True)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = float(point_size)

    # Fit camera to data
    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center().tolist())
    ctr.set_front([0, 0, -1])   # look along -Z
    ctr.set_up([0, -1, 0])      # Y down so Z is up
    ctr.set_zoom(0.35)

    print(f"label==1 points: {len(pts)}")
    vis.run()
    vis.destroy_window()


def to_xyz(arr: np.ndarray) -> np.ndarray:
    """
    Return an (N,3) xyz array from common shapes, without subsampling or normalization.
    - (N,>=3): use first 3 columns
    - (3,N): transpose to (N,3)
    - (1D): reshape to (-1,3) if divisible by 3
    - (3D): assume voxel grid; take all indices where value > threshold
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.shape[1] >= 3:
            xyz = a[:, :3]
        elif a.shape[0] >= 3:
            xyz = a[:3, :].T
        else:
            raise ValueError(f"2D array shape {a.shape} cannot provide 3 coordinates.")
    elif a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError("1D array length must be divisible by 3.")
        xyz = a.reshape(-1, 3)
    elif a.ndim == 3:
        idx = np.argwhere(a > voxel_threshold_for_grid).astype(np.float32)  # all occupied voxels
        xyz = idx
    else:
        raise ValueError(f"Unsupported array ndim={a.ndim}")
    return xyz.astype(np.float64, copy=False)

if __name__ == "__main__":
    file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_4.npy"
    # file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_op_4_cpnus_on_3d.npy"
    # file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_4_cpnus_on_3d.npy"
    # file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_4_cpnus0_2_on_3d.npy"
    # 1) Load
    arr = np.load(file_path, allow_pickle=False)
    print("Loaded shape:", arr.shape, "| dtype:", arr.dtype)

    #-------VISUALISE ONLY MASKED REGION----------------
    visualize_label1(arr, point_size=4.0)  # shows only label==1 region

    ##-------VISUALISE THE ENTIRE 3D IMAGE---------------
    # # 2) Convert to Nx3 (no GPU, no processing beyond basic shape handling)
    # xyz = to_xyz(arr)
    # print("Point count:", len(xyz))
    #
    # # 3) Build Open3D point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    #
    # # 4) Interactive view (adjust point size)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name=f"Point Cloud (Interactive): {len(xyz)}", width=1280, height=800, visible=True)
    # vis.add_geometry(pcd)
    # opt = vis.get_render_option()
    # opt.background_color = np.array([0, 0, 0])  # black background (optional)
    # opt.point_size = point_size
    # vis.run()          # rotate/zoom/pan with mouse; press H for help
    # vis.destroy_window()
