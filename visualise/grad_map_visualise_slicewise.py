# import numpy as np
# import open3d as o3d
# import os
# import matplotlib.pyplot as plt
# from matplotlib import cm, colors as mcolors
#
# # -------------------
# # Config
# # -------------------
# file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\point_cloud_3d_16.npy"
# point_size = 3.0                 # Open3D point size
# bin_size = 1.0                   # thickness of a Z slice
# min_points_per_slice = 1         # show even tiny slices
# recenter_geometry = True         # recenter each slice for VIEWING only
#
# # Gradient color scaling
# use_global_grad_scale = True     # True: same vmin/vmax across all slices; False: per-slice scaling
# manual_vmin = None               # set to a float to fix scale (overrides auto)
# manual_vmax = None
#
# # --- PNG saving ---
# png_dir = r"../output/3d/pc_test_slice/grad_view_slice_img"
# auto_save_png = True
# figsize = (6, 6)
# dpi = 200
# scatter_size = 6                 # matplotlib marker size for saved PNGs
# save_colorbar = True             # add colorbar to saved images
#
# # -------------------
# # Helpers
# # -------------------
# def load_xyz_grad(arr: np.ndarray):
#     """
#     Return xyz (Nx3 float64), grad (N float64).
#     Assumes columns [x,y,z,grad,label] or at least [x,y,z,grad].
#     Ignores label.
#     """
#     a = np.asarray(arr)
#     if a.ndim != 2 or a.shape[1] < 4:
#         raise ValueError(f"Expected N x 5 (or N x 4). Got {a.shape}.")
#     xyz = a[:, :3].astype(np.float64, copy=False)
#     grad = a[:, 3].astype(np.float64, copy=False)
#
#     m = np.isfinite(xyz).all(axis=1) & np.isfinite(grad)
#     return xyz[m], grad[m]
#
# def build_z_slices(xyz: np.ndarray, grad: np.ndarray, bin_size: float, min_pts: int):
#     z = xyz[:, 2]
#     zmin, zmax = float(z.min()), float(z.max())
#     edges = np.arange(zmin, zmax + bin_size, bin_size, dtype=np.float64)
#     if edges[-1] < zmax:
#         edges = np.append(edges, zmax)
#     centers = (edges[:-1] + edges[1:]) / 2.0
#     b = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
#
#     slices_pts, slices_grad, slice_centers = [], [], []
#     for i in range(len(edges) - 1):
#         mask = (b == i)
#         if mask.sum() >= min_pts:
#             slices_pts.append(xyz[mask])
#             slices_grad.append(grad[mask])
#             slice_centers.append(centers[i])
#     return slices_pts, slices_grad, slice_centers, edges
#
# def viridis_colors(values, vmin=None, vmax=None):
#     """
#     Map 1D array 'values' -> Nx3 RGB using viridis.
#     """
#     vals = np.asarray(values, dtype=np.float64)
#     if vmin is None:
#         vmin = float(np.min(vals))
#     if vmax is None:
#         vmax = float(np.max(vals))
#     # avoid zero range
#     if vmax <= vmin:
#         vmax = vmin + 1e-12
#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
#     cmap = cm.get_cmap('viridis')
#     rgba = cmap(norm(vals))  # Nx4
#     rgb = rgba[:, :3]
#     return rgb.astype(np.float64, copy=False), (vmin, vmax)
#
# def fit_camera_to(visualizer: o3d.visualization.Visualizer, geometry: o3d.geometry.Geometry3D, zoom=0.35):
#     ctr = visualizer.get_view_control()
#     bbox = geometry.get_axis_aligned_bounding_box()
#     ctr.set_lookat(bbox.get_center().tolist())
#     ctr.set_front([0, 0, -1])  # look along -Z
#     ctr.set_up([0, -1, 0])     # Y up
#     ctr.set_zoom(zoom)
#
# def save_slice_png(out_dir, idx1, xy, grad, vmin, vmax, dpi=200, figsize=(6,6), s=6, title=None, add_colorbar=True, xy_limits=None):
#     os.makedirs(out_dir, exist_ok=True)
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     ax = fig.add_subplot(111)
#     fig.patch.set_facecolor('white'); ax.set_facecolor('white')
#
#     sc = ax.scatter(xy[:,0], xy[:,1], s=s, c=grad, cmap='viridis', vmin=vmin, vmax=vmax, marker='.', linewidths=0)
#     if xy_limits is not None:
#         (xmin, xmax), (ymin, ymax) = xy_limits
#         ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
#
#     ax.set_aspect('equal', adjustable='box')
#     ax.axis('off')
#     if add_colorbar:
#         cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
#         cb.set_label("Gradient", rotation=90)
#
#     out_path = os.path.join(out_dir, f"sampled_points_slice_{idx1}.png")
#     plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
#     plt.close(fig)
#     return out_path
#
# # -------------------
# # Main
# # -------------------
# if __name__ == "__main__":
#     arr = np.load(file_path, allow_pickle=False)
#     xyz, grad = load_xyz_grad(arr)
#     if xyz.size == 0:
#         raise RuntimeError("No finite points found.")
#
#     print(f"Total points: {len(xyz)} | Z-range: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")
#
#     slices, slices_grad, centers, edges = build_z_slices(
#         xyz, grad, bin_size=bin_size, min_pts=min_points_per_slice
#     )
#     if not slices:
#         raise RuntimeError("No non-empty slices. Increase bin_size or lower min_points_per_slice.")
#
#     print(f"{len(slices)} non-empty slices (bin_size={bin_size}). Z spans [{edges[0]:.3f}, {edges[-1]:.3f}]")
#     print(f"First slice points: {len(slices[0])}, Z≈{centers[0]:.3f}")
#
#     # Global XY limits for consistent saved framing
#     xy_min = np.min(xyz[:, :2], axis=0)
#     xy_max = np.max(xyz[:, :2], axis=0)
#     xy_limits = ((xy_min[0], xy_max[0]), (xy_min[1], xy_max[1]))
#
#     # Decide vmin/vmax for coloring
#     if manual_vmin is not None and manual_vmax is not None:
#         global_vmin, global_vmax = float(manual_vmin), float(manual_vmax)
#     elif use_global_grad_scale:
#         global_vmin, global_vmax = float(np.min(grad)), float(np.max(grad))
#     else:
#         global_vmin = global_vmax = None  # per-slice scaling
#
#     # Open3D viewer init (first slice)
#     pcd = o3d.geometry.PointCloud()
#     first_pts = slices[0]
#     first_grad = slices_grad[0]
#
#     # Viewer uses recentered geometry (optional)
#     vpts = first_pts - first_pts.mean(axis=0, keepdims=True) if recenter_geometry and len(first_pts) else first_pts
#     pcd.points = o3d.utility.Vector3dVector(vpts)
#
#     # Color by gradient (viridis)
#     if use_global_grad_scale or (manual_vmin is not None and manual_vmax is not None):
#         colors_rgb, _ = viridis_colors(first_grad, vmin=global_vmin, vmax=global_vmax)
#     else:
#         colors_rgb, (global_vmin, global_vmax) = viridis_colors(first_grad)  # initialize from first slice
#     pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
#
#     # Axis & viewer
#     extent = (xyz.max(0) - xyz.min(0)).max()
#     axis_size = float(extent / 20.0) if np.isfinite(extent) and extent > 0 else 1.0
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
#
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name=f"Z-slice viewer (XY planes) | total points={len(xyz)}",
#                       width=1280, height=800, visible=True)
#     vis.add_geometry(pcd)
#     vis.add_geometry(axis)
#     opt = vis.get_render_option()
#     opt.background_color = np.array([0, 0, 0])   # black background in viewer
#     opt.point_size = float(point_size)
#
#     state = {"i": 0, "recenter": recenter_geometry, "psize": float(point_size)}
#
#     # Save PNG of first slice
#     if auto_save_png:
#         out = save_slice_png(
#             png_dir, 1, slices[0][:, :2], slices_grad[0],
#             vmin=(manual_vmin if manual_vmin is not None else global_vmin),
#             vmax=(manual_vmax if manual_vmax is not None else global_vmax),
#             dpi=dpi, figsize=figsize, s=scatter_size, title=f"Z≈{centers[0]:.3f}",
#             add_colorbar=save_colorbar, xy_limits=xy_limits
#         )
#         print(f"Saved PNG: {out}")
#
#     def set_slice(vis, idx):
#         idx = max(0, min(idx, len(slices) - 1))
#         state["i"] = idx
#         pts = slices[idx]
#         g = slices_grad[idx]
#
#         # Recenter for viewer only
#         vpts = pts - pts.mean(axis=0, keepdims=True) if state["recenter"] and len(pts) else pts
#         pcd.points = o3d.utility.Vector3dVector(vpts)
#
#         # Colors for this slice
#         vmin = manual_vmin if manual_vmin is not None else (global_vmin if use_global_grad_scale else None)
#         vmax = manual_vmax if manual_vmax is not None else (global_vmax if use_global_grad_scale else None)
#         cols, _ = viridis_colors(g, vmin=vmin, vmax=vmax)
#         pcd.colors = o3d.utility.Vector3dVector(cols)
#
#         vis.update_geometry(pcd)
#         fit_camera_to(vis, pcd, zoom=0.35)
#         vis.poll_events()
#         vis.update_renderer()
#
#         print(f"Slice {idx+1}/{len(slices)} | Z≈{centers[idx]:.3f} | points={len(pts)} "
#               f"| grad[vmin={vmin if vmin is not None else np.min(g):.3f}, vmax={vmax if vmax is not None else np.max(g):.3f}] "
#               f"| recentered={state['recenter']}")
#
#         # Save PNG
#         if auto_save_png:
#             out = save_slice_png(
#                 png_dir, idx + 1, pts[:, :2], g,
#                 vmin=(vmin if vmin is not None else np.min(g)),
#                 vmax=(vmax if vmax is not None else np.max(g)),
#                 dpi=dpi, figsize=figsize, s=scatter_size,
#                 title=f"Z≈{centers[idx]:.3f}", add_colorbar=save_colorbar, xy_limits=xy_limits
#             )
#             print(f"Saved PNG: {out}")
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
#     # Save ALL slices on 'S'
#     def cb_save_all(vis):
#         print(f"Saving all {len(slices)} slices to: {png_dir}")
#         if use_global_grad_scale and manual_vmin is None and manual_vmax is None:
#             vmin, vmax = global_vmin, global_vmax
#         else:
#             vmin = manual_vmin; vmax = manual_vmax
#         for i in range(len(slices)):
#             gi = slices_grad[i]
#             vi = vmin if vmin is not None else float(np.min(gi))
#             va = vmax if vmax is not None else float(np.max(gi))
#             save_slice_png(png_dir, i + 1, slices[i][:, :2], gi,
#                            vmin=vi, vmax=va, dpi=dpi, figsize=figsize, s=scatter_size,
#                            title=f"Z≈{centers[i]:.3f}", add_colorbar=save_colorbar, xy_limits=xy_limits)
#         print("Done saving all slices.")
#         return False
#
#     vis.register_key_callback(ord('K'), cb_next)
#     vis.register_key_callback(ord('J'), cb_prev)
#     vis.register_key_callback(GLFW_KEY_RIGHT, cb_next)
#     vis.register_key_callback(GLFW_KEY_LEFT, cb_prev)
#     vis.register_key_callback(ord('+'), cb_plus)
#     vis.register_key_callback(ord('='), cb_plus)
#     vis.register_key_callback(ord('-'), cb_minus)
#     vis.register_key_callback(ord('C'), cb_center_toggle)
#     vis.register_key_callback(ord('S'), cb_save_all)
#
#     print("\nControls:  K/→ next | J/← prev | +/- point size | C recenter toggle | S save all slices\n")
#     set_slice(vis, 0)
#     vis.run()
#     vis.destroy_window()

import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

# -------------------
# Config
# -------------------
file_path = r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_16.npy"
point_size = 3.0                 # Open3D point size
bin_size = 1.0                   # thickness of a Z slice
min_points_per_slice = 1         # show even tiny slices
recenter_geometry = True         # recenter each slice for VIEWING only

# Orientation controls
swap_xy = True                   # swap X and Y for display
flip_y  = True                   # flip Y vertically (fix upside-down)

# Gradient color scaling
use_global_grad_scale = True     # True: same vmin/vmax across all slices; False: per-slice scaling
manual_vmin = None               # set to a float to fix scale (overrides auto)
manual_vmax = None

# --- PNG saving ---
png_dir = r"../output/3d/pc_test_slice/sampled_points_slice_img"
auto_save_png = True
figsize = (6, 6)
dpi = 200
scatter_size = 6                 # matplotlib marker size for saved PNGs
save_colorbar = True             # add colorbar to saved images

# -------------------
# Helpers
# -------------------
def load_xyz_grad(arr: np.ndarray):
    """
    Return xyz (Nx3 float64), grad (N float64).
    Assumes columns [x,y,z,grad,label] or at least [x,y,z,grad].
    Ignores label.
    """
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] < 4:
        raise ValueError(f"Expected N x 5 (or N x 4). Got {a.shape}.")
    xyz = a[:, :3].astype(np.float64, copy=False)
    grad = a[:, 3].astype(np.float64, copy=False)

    m = np.isfinite(xyz).all(axis=1) & np.isfinite(grad)
    return xyz[m], grad[m]

def build_z_slices(xyz: np.ndarray, grad: np.ndarray, bin_size: float, min_pts: int):
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    edges = np.arange(zmin, zmax + bin_size, bin_size, dtype=np.float64)
    if edges[-1] < zmax:
        edges = np.append(edges, zmax)
    centers = (edges[:-1] + edges[1:]) / 2.0
    b = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)

    slices_pts, slices_grad, slice_centers = [], [], []
    for i in range(len(edges) - 1):
        mask = (b == i)
        if mask.sum() >= min_pts:
            slices_pts.append(xyz[mask])
            slices_grad.append(grad[mask])
            slice_centers.append(centers[i])
    return slices_pts, slices_grad, slice_centers, edges

def viridis_colors(values, vmin=None, vmax=None):
    """
    Map 1D array 'values' -> Nx3 RGB using viridis.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(vals))
    if vmax is None:
        vmax = float(np.max(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap('viridis')
    rgba = cmap(norm(vals))  # Nx4
    rgb = rgba[:, :3]
    return rgb.astype(np.float64, copy=False), (vmin, vmax)

def fit_camera_to(visualizer: o3d.visualization.Visualizer, geometry: o3d.geometry.Geometry3D, zoom=0.35):
    ctr = visualizer.get_view_control()
    bbox = geometry.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center().tolist())
    ctr.set_front([0, 0, -1])  # look along -Z
    ctr.set_up([0, 1, 0])      # Y up (positive)
    ctr.set_zoom(zoom)

def save_slice_png(out_dir, idx1, xy, grad, vmin, vmax,
                   dpi=200, figsize=(6,6), s=6, title=None,
                   add_colorbar=True, xy_limits=None, flip_y_display=False):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    sc = ax.scatter(xy[:,0], xy[:,1], s=s, c=grad, cmap='viridis',
                    vmin=vmin, vmax=vmax, marker='.', linewidths=0)

    if xy_limits is not None:
        (xmin, xmax), (ymin, ymax) = xy_limits
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    if flip_y_display:
        ax.invert_yaxis()  # correct upside-down in saved images

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if add_colorbar:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Gradient", rotation=90)

    out_path = os.path.join(out_dir, f"sampled_points_slice_{idx1}.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return out_path

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    arr = np.load(file_path, allow_pickle=False)
    xyz, grad = load_xyz_grad(arr)
    if xyz.size == 0:
        raise RuntimeError("No finite points found.")

    print(f"Total points: {len(xyz)} | Z-range: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

    slices, slices_grad, centers, edges = build_z_slices(
        xyz, grad, bin_size=bin_size, min_pts=min_points_per_slice
    )
    if not slices:
        raise RuntimeError("No non-empty slices. Increase bin_size or lower min_points_per_slice.")

    print(f"{len(slices)} non-empty slices (bin_size={bin_size}). Z spans [{edges[0]:.3f}, {edges[-1]:.3f}]")
    print(f"First slice points: {len(slices[0])}, Z≈{centers[0]:.3f}")

    # ---- Global XY limits (use swapped axes if enabled) ----
    if swap_xy:
        xy_min = np.min(xyz[:, [1, 0]], axis=0)
        xy_max = np.max(xyz[:, [1, 0]], axis=0)
    else:
        xy_min = np.min(xyz[:, :2], axis=0)
        xy_max = np.max(xyz[:, :2], axis=0)
    xy_limits = ((xy_min[0], xy_max[0]), (xy_min[1], xy_max[1]))

    # Decide vmin/vmax for coloring
    if manual_vmin is not None and manual_vmax is not None:
        global_vmin, global_vmax = float(manual_vmin), float(manual_vmax)
    elif use_global_grad_scale:
        global_vmin, global_vmax = float(np.min(grad)), float(np.max(grad))
    else:
        global_vmin = global_vmax = None  # per-slice scaling

    # ----- Open3D viewer init (first slice) -----
    pcd = o3d.geometry.PointCloud()
    first_pts = slices[0].copy()
    first_grad = slices_grad[0]

    # Swap X/Y for viewer
    if swap_xy:
        first_pts = first_pts[:, [1, 0, 2]]  # (y, x, z)
    # Flip Y for viewer to fix upside-down
    if flip_y:
        first_pts[:, 1] *= -1.0

    # Recenter for viewer only
    vpts = first_pts - first_pts.mean(axis=0, keepdims=True) if recenter_geometry and len(first_pts) else first_pts
    pcd.points = o3d.utility.Vector3dVector(vpts)

    # Color by gradient (viridis)
    if use_global_grad_scale or (manual_vmin is not None and manual_vmax is not None):
        colors_rgb, _ = viridis_colors(first_grad, vmin=global_vmin, vmax=global_vmax)
    else:
        colors_rgb, (global_vmin, global_vmax) = viridis_colors(first_grad)  # initialize from first slice
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    # Axis & viewer
    extent = (xyz.max(0) - xyz.min(0)).max()
    axis_size = float(extent / 20.0) if np.isfinite(extent) and extent > 0 else 1.0
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Z-slice viewer (XY planes) | total points={len(xyz)}",
                      width=1280, height=800, visible=True)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])   # black background in viewer
    opt.point_size = float(point_size)

    state = {"i": 0, "recenter": recenter_geometry, "psize": float(point_size)}

    # Save PNG of first slice (swap XY for image; flip Y via axis inversion)
    if auto_save_png:
        xy0 = slices[0][:, [1, 0]] if swap_xy else slices[0][:, :2]
        out = save_slice_png(
            png_dir, 1, xy0, slices_grad[0],
            vmin=(manual_vmin if manual_vmin is not None else global_vmin),
            vmax=(manual_vmax if manual_vmax is not None else global_vmax),
            dpi=dpi, figsize=figsize, s=scatter_size, title=f"Z≈{centers[0]:.3f}",
            add_colorbar=save_colorbar, xy_limits=xy_limits, flip_y_display=flip_y
        )
        print(f"Saved PNG: {out}")

    def set_slice(vis, idx):
        idx = max(0, min(idx, len(slices) - 1))
        state["i"] = idx
        pts = slices[idx].copy()
        g = slices_grad[idx]

        # Swap X/Y for viewer
        if swap_xy:
            pts = pts[:, [1, 0, 2]]
        # Flip Y for viewer
        if flip_y:
            pts[:, 1] *= -1.0

        # Recenter for viewer only
        vpts = pts - pts.mean(axis=0, keepdims=True) if state["recenter"] and len(pts) else pts
        pcd.points = o3d.utility.Vector3dVector(vpts)

        # Colors for this slice
        vmin = manual_vmin if manual_vmin is not None else (global_vmin if use_global_grad_scale else None)
        vmax = manual_vmax if manual_vmax is not None else (global_vmax if use_global_grad_scale else None)
        cols, _ = viridis_colors(g, vmin=vmin, vmax=vmax)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        vis.update_geometry(pcd)
        fit_camera_to(vis, pcd, zoom=0.35)
        vis.poll_events()
        vis.update_renderer()

        print(f"Slice {idx+1}/{len(slices)} | Z≈{centers[idx]:.3f} | points={len(pts)} "
              f"| grad[vmin={vmin if vmin is not None else np.min(g):.3f}, vmax={vmax if vmax is not None else np.max(g):.3f}] "
              f"| recentered={state['recenter']}")

        # Save PNG (swap XY for image; flip Y via axis inversion)
        if auto_save_png:
            xy = slices[idx][:, [1, 0]] if swap_xy else slices[idx][:, :2]
            out = save_slice_png(
                png_dir, idx + 1, xy, g,
                vmin=(vmin if vmin is not None else np.min(g)),
                vmax=(vmax if vmax is not None else np.max(g)),
                dpi=dpi, figsize=figsize, s=scatter_size,
                title=f"Z≈{centers[idx]:.3f}", add_colorbar=save_colorbar,
                xy_limits=xy_limits, flip_y_display=flip_y
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
        if use_global_grad_scale and manual_vmin is None and manual_vmax is None:
            vmin, vmax = global_vmin, global_vmax
        else:
            vmin = manual_vmin; vmax = manual_vmax
        for i in range(len(slices)):
            gi = slices_grad[i]
            vi = vmin if vmin is not None else float(np.min(gi))
            va = vmax if vmax is not None else float(np.max(gi))
            xy = slices[i][:, [1, 0]] if swap_xy else slices[i][:, :2]
            save_slice_png(png_dir, i + 1, xy, gi,
                           vmin=vi, vmax=va, dpi=dpi, figsize=figsize, s=scatter_size,
                           title=f"Z≈{centers[i]:.3f}", add_colorbar=save_colorbar,
                           xy_limits=xy_limits, flip_y_display=flip_y)
        print("Done saving all slices.")
        return False

    vis.register_key_callback(ord('K'), cb_next)
    vis.register_key_callback(ord('J'), cb_prev)
    vis.register_key_callback(GLFW_KEY_RIGHT, cb_next)
    vis.register_key_callback(GLFW_KEY_LEFT, cb_prev)
    vis.register_key_callback(ord('+'), cb_plus)
    vis.register_key_callback(ord('='), cb_plus)
    vis.register_key_callback(ord('-'), cb_minus)
    vis.register_key_callback(ord('C'), cb_center_toggle)
    vis.register_key_callback(ord('S'), cb_save_all)

    print("\nControls:  K/→ next | J/← prev | +/- point size | C recenter toggle | S save all slices\n")
    set_slice(vis, 0)
    vis.run()
    vis.destroy_window()
