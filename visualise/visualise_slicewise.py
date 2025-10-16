import numpy as np
import open3d as o3d

# -------------------
# Config
# -------------------
file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_15.npy"
point_size = 3.0                 # thicker points to be visible immediately
bin_size = 1.0                   # thickness of a Z slice
min_points_per_slice = 1         # show even tiny slices
recenter_geometry = True         # center each slice at its own centroid

# -------------------
# Helpers
# -------------------
def to_xyz(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        xyz = a[:, :3] if a.shape[1] >= 3 else a[:3, :].T
    elif a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError("1D array length must be divisible by 3.")
        xyz = a.reshape(-1, 3)
    elif a.ndim == 3:
        # assume voxel grid -> indices (z,y,x) -> reorder to (x,y,z)
        idx = np.argwhere(a > 0).astype(np.float64)
        xyz = np.stack([idx[:, 2], idx[:, 1], idx[:, 0]], axis=1)
    else:
        raise ValueError(f"Unsupported array ndim={a.ndim}")
    m = np.isfinite(xyz).all(axis=1)
    return xyz[m].astype(np.float64, copy=False)

def build_z_slices(xyz: np.ndarray, bin_size: float, min_pts: int):
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    edges = np.arange(zmin, zmax + bin_size, bin_size, dtype=np.float64)
    if edges[-1] < zmax:
        edges = np.append(edges, zmax)
    centers = (edges[:-1] + edges[1:]) / 2.0
    b = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    slices, slice_centers = [], []
    for i in range(len(edges) - 1):
        mask = (b == i)
        if mask.sum() >= min_pts:
            slices.append(xyz[mask])
            slice_centers.append(centers[i])
    return slices, slice_centers, edges

def fit_camera_to(visualizer: o3d.visualization.Visualizer, geometry: o3d.geometry.Geometry3D, zoom=0.35):
    ctr = visualizer.get_view_control()
    bbox = geometry.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center().tolist())
    # look along -Z with Y up; tweak if your coordinate system differs
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(zoom)  # ~0.35 is a good default in Open3D

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    arr = np.load(file_path, allow_pickle=False)
    xyz = to_xyz(arr)
    if xyz.size == 0:
        raise RuntimeError("No finite points found.")

    print(f"Total points: {len(xyz)} | Z-range: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

    slices, centers, edges = build_z_slices(xyz, bin_size=bin_size, min_pts=min_points_per_slice)
    if not slices:
        raise RuntimeError("No non-empty slices. Increase bin_size or lower min_points_per_slice.")

    print(f"{len(slices)} non-empty slices (bin_size={bin_size}). Z spans [{edges[0]:.3f}, {edges[-1]:.3f}]")
    print(f"First slice points: {len(slices[0])}, Z≈{centers[0]:.3f}")

    # Geometry
    pcd = o3d.geometry.PointCloud()
    show_pts = slices[0]
    if recenter_geometry:
        show_pts = show_pts - show_pts.mean(axis=0, keepdims=True)
    pcd.points = o3d.utility.Vector3dVector(show_pts)
    pcd.paint_uniform_color([1.0, 1.0, 1.0])  # white points

    # Axis simply for orientation (small)
    extent = (xyz.max(0) - xyz.min(0)).max()
    axis_size = float(extent / 20.0) if np.isfinite(extent) and extent > 0 else 1.0
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Z-slice viewer (XY planes) | total points={len(xyz)}", width=1280, height=800, visible=True)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = float(point_size)

    state = {"i": 0, "recenter": recenter_geometry, "psize": float(point_size)}

    def set_slice(vis, idx):
        idx = max(0, min(idx, len(slices) - 1))
        state["i"] = idx
        pts = slices[idx]
        if state["recenter"] and len(pts):
            pts = pts - pts.mean(axis=0, keepdims=True)
        pcd.points = o3d.utility.Vector3dVector(pts)
        vis.update_geometry(pcd)
        fit_camera_to(vis, pcd, zoom=0.35)
        vis.poll_events()
        vis.update_renderer()
        print(f"Slice {idx+1}/{len(slices)} | Z≈{centers[idx]:.3f} | points={len(pts)} | recentered={state['recenter']}")

    # Key callbacks
    GLFW_KEY_LEFT, GLFW_KEY_RIGHT = 263, 262
    def cb_next(vis): set_slice(vis, state["i"] + 1); return False
    def cb_prev(vis): set_slice(vis, state["i"] - 1); return False
    def cb_plus(vis):
        state["psize"] = min(state["psize"] + 1.0, 15.0)
        vis.get_render_option().point_size = state["psize"]; return False
    def cb_minus(vis):
        state["psize"] = max(state["psize"] - 1.0, 1.0)
        vis.get_render_option().point_size = state["psize"]; return False
    def cb_center_toggle(vis):
        state["recenter"] = not state["recenter"]
        set_slice(vis, state["i"]); return False

    vis.register_key_callback(ord('K'), cb_next)
    vis.register_key_callback(ord('J'), cb_prev)
    vis.register_key_callback(GLFW_KEY_RIGHT, cb_next)
    vis.register_key_callback(GLFW_KEY_LEFT, cb_prev)
    vis.register_key_callback(ord('+'), cb_plus)
    vis.register_key_callback(ord('='), cb_plus)  # some keyboards
    vis.register_key_callback(ord('-'), cb_minus)
    vis.register_key_callback(ord('C'), cb_center_toggle)

    print("\nControls:  K/→ next | J/← prev | +/- point size | C recenter toggle | H help\n")
    set_slice(vis, 0)
    vis.run()
    vis.destroy_window()
