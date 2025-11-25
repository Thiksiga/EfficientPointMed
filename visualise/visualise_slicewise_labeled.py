# import numpy as np
# import open3d as o3d
#
# # -------------------
# # Config
# # -------------------
# file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_15.npy"
# point_size = 3.0                 # thicker points to be visible immediately
# bin_size = 1.0                   # thickness of a Z slice
# min_points_per_slice = 1         # show even tiny slices
# recenter_geometry = True         # center each slice at its own centroid
#
# # -------------------
# # Helpers
# # -------------------
# def to_xyz_labels(arr: np.ndarray):
#     """
#     Returns xyz (float64, Nx3) and labels (None or int64, shape N).
#     Assumes N x 5 with [x,y,z,grad,label] when available.
#     Supports:
#       - 2D arrays: Nx{3,4,5,...}
#       - 1D arrays length divisible by 3 (no labels)
#       - 3D voxel mask/volume -> extracts coordinates, no labels
#     """
#     a = np.asarray(arr)
#     labels = None
#
#     if a.ndim == 2:
#         if a.shape[1] >= 3:
#             xyz = a[:, :3].astype(np.float64, copy=False)
#             if a.shape[1] >= 5:
#                 labels = a[:, -1]
#             elif a.shape[1] == 4:
#                 # If it's [x,y,z,label] use last col as label
#                 labels = a[:, -1]
#         else:
#             # transpose fallback if input is 3xN
#             xyz = a[:3, :].T.astype(np.float64, copy=False)
#             labels = None
#
#     elif a.ndim == 1:
#         if a.size % 3 != 0:
#             raise ValueError("1D array length must be divisible by 3.")
#         xyz = a.reshape(-1, 3).astype(np.float64, copy=False)
#
#     elif a.ndim == 3:
#         # assume voxel grid -> indices (z,y,x) -> reorder to (x,y,z)
#         idx = np.argwhere(a > 0).astype(np.float64)
#         xyz = np.stack([idx[:, 2], idx[:, 1], idx[:, 0]], axis=1)
#         labels = None
#     else:
#         raise ValueError(f"Unsupported array ndim={a.ndim}")
#
#     # Clean NaNs/Infs
#     m = np.isfinite(xyz).all(axis=1)
#     xyz = xyz[m]
#     if labels is not None:
#         labels = labels[m]
#         # Cast labels to int if they look binary/integers
#         if np.issubdtype(labels.dtype, np.floating):
#             labels = (labels > 0.5).astype(np.int64)
#         else:
#             labels = labels.astype(np.int64, copy=False)
#     return xyz, labels
#
# def build_z_slices(xyz: np.ndarray, labels, bin_size: float, min_pts: int):
#     z = xyz[:, 2]
#     zmin, zmax = float(z.min()), float(z.max())
#     edges = np.arange(zmin, zmax + bin_size, bin_size, dtype=np.float64)
#     if edges[-1] < zmax:
#         edges = np.append(edges, zmax)
#     centers = (edges[:-1] + edges[1:]) / 2.0
#     b = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
#     slices_pts, slices_labels, slice_centers = [], [], []
#     for i in range(len(edges) - 1):
#         mask = (b == i)
#         if mask.sum() >= min_pts:
#             slices_pts.append(xyz[mask])
#             if labels is not None:
#                 slices_labels.append(labels[mask])
#             else:
#                 slices_labels.append(None)
#             slice_centers.append(centers[i])
#     return slices_pts, slices_labels, slice_centers, edges
#
# def colors_for_labels(lbls, n_points):
#     """
#     Returns Nx3 colors (float) with label==1 in red, others white.
#     If lbls is None, returns all white.
#     """
#     cols = np.ones((n_points, 3), dtype=np.float64)  # white
#     if lbls is not None:
#         m1 = (lbls == 1)
#         cols[m1] = np.array([1.0, 0.0, 0.0])  # red
#     return cols
#
# def fit_camera_to(visualizer: o3d.visualization.Visualizer, geometry: o3d.geometry.Geometry3D, zoom=0.35):
#     ctr = visualizer.get_view_control()
#     bbox = geometry.get_axis_aligned_bounding_box()
#     ctr.set_lookat(bbox.get_center().tolist())
#     # look along -Z with Y up; tweak if your coordinate system differs
#     ctr.set_front([0, 0, -1])
#     ctr.set_up([0, -1, 0])
#     ctr.set_zoom(zoom)  # ~0.35 is a good default in Open3D
#
# # -------------------
# # Main
# # -------------------
# if __name__ == "__main__":
#     arr = np.load(file_path, allow_pickle=False)
#     xyz, labels = to_xyz_labels(arr)
#     if xyz.size == 0:
#         raise RuntimeError("No finite points found.")
#
#     print(f"Total points: {len(xyz)} | Z-range: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")
#
#     slices, slices_lbls, centers, edges = build_z_slices(
#         xyz, labels, bin_size=bin_size, min_pts=min_points_per_slice
#     )
#     if not slices:
#         raise RuntimeError("No non-empty slices. Increase bin_size or lower min_points_per_slice.")
#
#     print(f"{len(slices)} non-empty slices (bin_size={bin_size}). Z spans [{edges[0]:.3f}, {edges[-1]:.3f}]")
#     print(f"First slice points: {len(slices[0])}, Z≈{centers[0]:.3f}")
#
#     # Geometry for first slice
#     pcd = o3d.geometry.PointCloud()
#     show_pts = slices[0]
#     show_lbls = slices_lbls[0]
#     if recenter_geometry and len(show_pts):
#         show_pts = show_pts - show_pts.mean(axis=0, keepdims=True)
#     pcd.points = o3d.utility.Vector3dVector(show_pts)
#
#     # Set per-point colors: label==1 -> red, else white
#     cols = colors_for_labels(show_lbls, len(show_pts))
#     pcd.colors = o3d.utility.Vector3dVector(cols)
#
#     # Axis for orientation
#     extent = (xyz.max(0) - xyz.min(0)).max()
#     axis_size = float(extent / 20.0) if np.isfinite(extent) and extent > 0 else 1.0
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
#
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name=f"Z-slice viewer (XY planes) | total points={len(xyz)}", width=1280, height=800, visible=True)
#     vis.add_geometry(pcd)
#     vis.add_geometry(axis)
#     opt = vis.get_render_option()
#     opt.background_color = np.array([0, 0, 0])
#     opt.point_size = float(point_size)
#
#     state = {"i": 0, "recenter": recenter_geometry, "psize": float(point_size)}
#
#     def set_slice(vis, idx):
#         idx = max(0, min(idx, len(slices) - 1))
#         state["i"] = idx
#         pts = slices[idx]
#         lbls = slices_lbls[idx]
#         if state["recenter"] and len(pts):
#             pts = pts - pts.mean(axis=0, keepdims=True)
#         pcd.points = o3d.utility.Vector3dVector(pts)
#         # update colors for this slice
#         pcd.colors = o3d.utility.Vector3dVector(colors_for_labels(lbls, len(pts)))
#         vis.update_geometry(pcd)
#         fit_camera_to(vis, pcd, zoom=0.35)
#         vis.poll_events()
#         vis.update_renderer()
#         n_red = int((lbls == 1).sum()) if lbls is not None else 0
#         np.save()
#         print(f"Slice {idx+1}/{len(slices)} | Z≈{centers[idx]:.3f} | points={len(pts)} "
#               f"| label1={n_red} | recentered={state['recenter']}")
#
#     # Key callbacks
#     GLFW_KEY_LEFT, GLFW_KEY_RIGHT = 263, 262
#     def cb_next(vis): set_slice(vis, state["i"] + 1); return False
#     def cb_prev(vis): set_slice(vis, state["i"] - 1); return False
#     def cb_plus(vis):
#         state["psize"] = min(state["psize"] + 1.0, 15.0)
#         vis.get_render_option().point_size = state["psize"]; return False
#     def cb_minus(vis):
#         state["psize"] = max(state["psize"] - 1.0, 1.0)
#         vis.get_render_option().point_size = state["psize"]; return False
#     def cb_center_toggle(vis):
#         state["recenter"] = not state["recenter"]
#         set_slice(vis, state["i"]); return False
#
#     vis.register_key_callback(ord('K'), cb_next)
#     vis.register_key_callback(ord('J'), cb_prev)
#     vis.register_key_callback(GLFW_KEY_RIGHT, cb_next)
#     vis.register_key_callback(GLFW_KEY_LEFT, cb_prev)
#     vis.register_key_callback(ord('+'), cb_plus)
#     vis.register_key_callback(ord('='), cb_plus)  # some keyboards
#     vis.register_key_callback(ord('-'), cb_minus)
#     vis.register_key_callback(ord('C'), cb_center_toggle)
#
#     print("\nControls:  K/→ next | J/← prev | +/- point size | C recenter toggle | H help\n")
#     set_slice(vis, 0)
#     vis.run()
#     vis.destroy_window()


import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# -------------------
# Config
# -------------------
file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_18_kn2.npy"
point_size = 3.0                 # Open3D point size
bin_size = 1.0                   # thickness of a Z slice
min_points_per_slice = 1         # show even tiny slices
recenter_geometry = True         # recenter each slice for VIEWING only

# --- PNG saving (as requested) ---
png_dir = r"../output/3d/pc_test_slice/pointcloud_unsampled_img_18_kn2"
auto_save_png = True
figsize = (6, 6)                 # inches
dpi = 200
scatter_size = 6                 # matplotlib marker size for saved PNGs

# -------------------
# Helpers
# -------------------
def to_xyz_labels(arr: np.ndarray):
    """
    Returns xyz (float64, Nx3) and labels (None or int64, shape N).
    Assumes N x 5 with [x,y,z,grad,label] when available.
    """
    a = np.asarray(arr)
    labels = None

    if a.ndim == 2:
        if a.shape[1] >= 3:
            xyz = a[:, :3].astype(np.float64, copy=False)
            if a.shape[1] >= 5:
                labels = a[:, -1]
            elif a.shape[1] == 4:
                labels = a[:, -1]
        else:
            xyz = a[:3, :].T.astype(np.float64, copy=False)
            labels = None
    elif a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError("1D array length must be divisible by 3.")
        xyz = a.reshape(-1, 3).astype(np.float64, copy=False)
    elif a.ndim == 3:
        # assume voxel grid -> indices (z,y,x) -> reorder to (x,y,z)
        idx = np.argwhere(a > 0).astype(np.float64)
        xyz = np.stack([idx[:, 2], idx[:, 1], idx[:, 0]], axis=1)
        labels = None
    else:
        raise ValueError(f"Unsupported array ndim={a.ndim}")

    m = np.isfinite(xyz).all(axis=1)
    xyz = xyz[m]
    if labels is not None:
        labels = labels[m]
        if np.issubdtype(labels.dtype, np.floating):
            labels = (labels > 0.5).astype(np.int64)
        else:
            labels = labels.astype(np.int64, copy=False)
    return xyz, labels

def build_z_slices(xyz: np.ndarray, labels, bin_size: float, min_pts: int):
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    edges = np.arange(zmin, zmax + bin_size, bin_size, dtype=np.float64)
    if edges[-1] < zmax:
        edges = np.append(edges, zmax)
    centers = (edges[:-1] + edges[1:]) / 2.0
    b = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    slices_pts, slices_lbls, slice_centers = [], [], []
    for i in range(len(edges) - 1):
        mask = (b == i)
        if mask.sum() >= min_pts:
            slices_pts.append(xyz[mask])
            slices_lbls.append(labels[mask] if labels is not None else None)
            slice_centers.append(centers[i])
    return slices_pts, slices_lbls, slice_centers, edges

def colors_for_labels(lbls, n_points):
    cols = np.ones((n_points, 3), dtype=np.float64)  # white for viewer
    if lbls is not None:
        m1 = (lbls == 1)
        cols[m1] = np.array([1.0, 0.0, 0.0])  # red
    return cols

def fit_camera_to(visualizer: o3d.visualization.Visualizer, geometry: o3d.geometry.Geometry3D, zoom=0.35):
    ctr = visualizer.get_view_control()
    bbox = geometry.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center().tolist())
    ctr.set_front([0, 0, -1])  # look along -Z
    ctr.set_up([0, -1, 0])     # Y up
    ctr.set_zoom(zoom)

# ---------- PNG saving helper (global XY frame, white bg, black pts, red label==1) ----------
def save_slice_png_from_xy(out_dir, slice_index_one_based, xy, labels=None,
                           xy_limits=None, dpi=200, figsize=(6,6), s=6, title=None):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # All points in black
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c='k', marker='.', linewidths=0)

    # Overlay label==1 in red
    if labels is not None and len(labels) == len(xy):
        m1 = (labels == 1)
        if np.any(m1):
            ax.scatter(xy[m1, 0], xy[m1, 1], s=s, c='r', marker='.', linewidths=0)

    if xy_limits is not None:
        (xmin, xmax), (ymin, ymax) = xy_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # EXACT requested path & filename pattern
    out_path = os.path.join(out_dir, f"sampled_points_slice_{slice_index_one_based}.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return out_path

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    arr = np.load(file_path, allow_pickle=False)
    xyz, labels = to_xyz_labels(arr)
    if xyz.size == 0:
        raise RuntimeError("No finite points found.")

    print(f"Total points: {len(xyz)} | Z-range: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

    slices, slices_lbls, centers, edges = build_z_slices(
        xyz, labels, bin_size=bin_size, min_pts=min_points_per_slice
    )
    if not slices:
        raise RuntimeError("No non-empty slices. Increase bin_size or lower min_points_per_slice.")

    print(f"{len(slices)} non-empty slices (bin_size={bin_size}). Z spans [{edges[0]:.3f}, {edges[-1]:.3f}]")
    print(f"First slice points: {len(slices[0])}, Z≈{centers[0]:.3f}")

    # Global XY limits so saved PNGs share same frame
    xy_min = np.min(xyz[:, :2], axis=0)
    xy_max = np.max(xyz[:, :2], axis=0)
    xy_limits = ((xy_min[0], xy_max[0]), (xy_min[1], xy_max[1]))

    # Open3D geometry for first slice (viewer)
    pcd = o3d.geometry.PointCloud()
    show_pts = slices[0]
    show_lbls = slices_lbls[0]
    vpts = show_pts - show_pts.mean(axis=0, keepdims=True) if recenter_geometry and len(show_pts) else show_pts
    pcd.points = o3d.utility.Vector3dVector(vpts)
    pcd.colors = o3d.utility.Vector3dVector(colors_for_labels(show_lbls, len(vpts)))

    # Axis for orientation
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

    # Save first slice PNG immediately (1-based index)
    if auto_save_png:
        p = save_slice_png_from_xy(
            png_dir, 1, slices[0][:, :2], slices_lbls[0],
            xy_limits=xy_limits, dpi=dpi, figsize=figsize, s=scatter_size,
            title=f"Z≈{centers[0]:.3f}"
        )
        print(f"Saved PNG: {p}")

    def set_slice(vis, idx):
        idx = max(0, min(idx, len(slices) - 1))
        state["i"] = idx
        pts = slices[idx]
        lbls = slices_lbls[idx]

        # Update viewer (optionally recentered)
        vpts = pts - pts.mean(axis=0, keepdims=True) if state["recenter"] and len(pts) else pts
        pcd.points = o3d.utility.Vector3dVector(vpts)
        pcd.colors = o3d.utility.Vector3dVector(colors_for_labels(lbls, len(vpts)))

        vis.update_geometry(pcd)
        fit_camera_to(vis, pcd, zoom=0.35)
        vis.poll_events()
        vis.update_renderer()

        n_red = int((lbls == 1).sum()) if lbls is not None else 0
        print(f"Slice {idx+1}/{len(slices)} | Z≈{centers[idx]:.3f} | points={len(pts)} "
              f"| label1={n_red} | recentered={state['recenter']}")

        # Save PNG for this slice in requested folder/pattern (1-based index)
        if auto_save_png:
            out = save_slice_png_from_xy(
                png_dir, idx + 1, pts[:, :2], lbls,
                xy_limits=xy_limits, dpi=dpi, figsize=figsize, s=scatter_size,
                title=f"Z≈{centers[idx]:.3f}"
            )
            print(f"Saved PNG: {out}")

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

    # Save ALL slices on 'S'
    def cb_save_all(vis):
        print(f"Saving all {len(slices)} slices to: {png_dir}")
        for i in range(len(slices)):
            save_slice_png_from_xy(
                png_dir, i + 1, slices[i][:, :2], slices_lbls[i],
                xy_limits=xy_limits, dpi=dpi, figsize=figsize, s=scatter_size,
                title=f"Z≈{centers[i]:.3f}"
            )
        print("Done saving all slices.")
        return False

    vis.register_key_callback(ord('K'), cb_next)
    vis.register_key_callback(ord('J'), cb_prev)
    vis.register_key_callback(GLFW_KEY_RIGHT, cb_next)
    vis.register_key_callback(GLFW_KEY_LEFT, cb_prev)
    vis.register_key_callback(ord('+'), cb_plus)
    vis.register_key_callback(ord('='), cb_plus)  # some keyboards
    vis.register_key_callback(ord('-'), cb_minus)
    vis.register_key_callback(ord('C'), cb_center_toggle)
    vis.register_key_callback(ord('S'), cb_save_all)

    print("\nControls:  K/→ next | J/← prev | +/- point size | C recenter toggle | S save all slices\n")
    set_slice(vis, 0)
    vis.run()
    vis.destroy_window()

