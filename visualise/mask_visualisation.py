import numpy as np
import open3d as o3d

def pc_from_edges_slice(edges_masked2d: np.ndarray,
                        magnitude2d: np.ndarray,
                        mask2d: np.ndarray,
                        y_idx: int) -> np.ndarray:
    """
    Build a 3D per-slice point cloud from a 2D edge map on slice y_idx.
    Input arrays are shape (x, z) for that slice. Output is (N,5): (x, y, z, grad, label).
    """
    # edge_points columns: [x, z] because np.where on (rows=x, cols=z)
    edge_points = np.column_stack(np.where(edges_masked2d > 0))
    if edge_points.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    x = edge_points[:, 0]
    z = edge_points[:, 1]
    y = np.full_like(x, fill_value=y_idx)

    grad  = magnitude2d[x, z]
    label = mask2d[x, z]           # 0/1 (masked region = 1)

    pc = np.column_stack([x, y, z, grad, label]).astype(np.float32, copy=False)
    return pc


def mask_slice_points_xyz(mask_xzy: np.ndarray, y_idx: int) -> np.ndarray:
    """
    From a 3D mask shaped (x, z, y), extract the y_idx slice and return (N,3) XYZ points as (x, y, z).
    """
    if not (0 <= y_idx < mask_xzy.shape[2]):
        raise IndexError(f"y_idx {y_idx} out of range for mask shape {mask_xzy.shape} (x,z,y)")

    slice2d = mask_xzy[:, :, y_idx]               # (x, z)
    idx = np.column_stack(np.where(slice2d > 0))  # [x, z]
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    x = idx[:, 0]
    z = idx[:, 1]
    y = np.full_like(x, fill_value=y_idx)
    return np.stack([x, y, z], axis=1).astype(np.float64, copy=False)


def visualize_mask_and_pc_slice(mask_xzy: np.ndarray,
                                pc_slice_xyzgl: np.ndarray,
                                y_idx: int,
                                point_size_mask: float = 2.0,
                                point_size_pc: float = 2.5):
    """
    Show both:
      - mask slice points (green)
      - point cloud slice points colored by label (red=label1, cyan=label0)
    """
    # 1) Mask slice → (N,3)
    mask_pts = mask_slice_points_xyz(mask_xzy, y_idx)

    # 2) PC slice → (N,3) + colors by label
    if pc_slice_xyzgl.ndim != 2 or pc_slice_xyzgl.shape[1] < 5:
        raise ValueError(f"pc_slice_xyzgl must be (N,5): (x,y,z,grad,label). Got {pc_slice_xyzgl.shape}")

    pc_xyz = pc_slice_xyzgl[:, :3].astype(np.float64, copy=False)
    pc_lab = pc_slice_xyzgl[:, 4] >= 0.5

    pc_colors = np.tile([0.3, 0.85, 1.0], (pc_xyz.shape[0], 1))  # cyan for label==0
    pc_colors[pc_lab] = [1.0, 0.2, 0.2]                          # red for label==1

    # 3) Build Open3D geometries
    geoms = []

    pcd_mask = o3d.geometry.PointCloud()
    if mask_pts.size:
        pcd_mask.points = o3d.utility.Vector3dVector(mask_pts)
        pcd_mask.paint_uniform_color([0.0, 1.0, 0.0])            # green
        geoms.append(pcd_mask)

    pcd_pc = o3d.geometry.PointCloud()
    if pc_xyz.size:
        pcd_pc.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd_pc.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64, copy=False))
        geoms.append(pcd_pc)

    if not geoms:
        print("Nothing to show: both mask slice and pc slice are empty.")
        return

    # Axis sized to the scene
    all_pts = np.vstack([g.points.__array__() for g in geoms])
    extent = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(max(extent / 20.0, 1.0)),
                                                             origin=[0, 0, 0])

    # 4) Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Slice y={y_idx} | mask (green) + pc (red/cyan)", width=1280, height=800, visible=True)
    for g in geoms:
        vis.add_geometry(g)
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = float(point_size_pc)  # start from PC size
    # set mask point size separately by duplicating render option is not supported; use a single size

    # Fit camera to all points
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(all_pts))
    ctr = vis.get_view_control()
    ctr.set_lookat(bbox.get_center().tolist())
    ctr.set_front([0, -1, 0])  # look along -Y so you face the slice plane
    ctr.set_up([0, 0, 1])      # Z up
    ctr.set_zoom(0.35)

    print(f"mask points: {len(mask_pts)} | pc points: {len(pc_xyz)} (label1={int(pc_lab.sum())}, label0={int((~pc_lab).sum())})")
    vis.run()
    vis.destroy_window()

import argparse
import numpy as np

def pc_slice_from_full_pc(pc_xyzgl: np.ndarray, y_idx: int, round_to_voxel: bool = True) -> np.ndarray:
    """
    Select rows from a full point cloud (N,5) where y == y_idx (after rounding if requested).
    Returns a (M,5) array (x,y,z,grad,label).
    """
    if pc_xyzgl.ndim != 2 or pc_xyzgl.shape[1] < 5:
        raise ValueError(f"Expected (N,5) array [x,y,z,grad,label]; got {pc_xyzgl.shape}")

    y = pc_xyzgl[:, 1]
    yv = np.rint(y).astype(int) if round_to_voxel else y.astype(int)
    mask = (yv == int(y_idx))
    return pc_xyzgl[mask]

def main():
    parser = argparse.ArgumentParser(description="Visualize mask (x,z,y) slice with point-cloud labels.")
    parser.add_argument("--mask", required=True, help="Path to 3D mask .npy (shape (x,z,y), e.g., (170,352,370))")
    parser.add_argument("--pc",   required=True, help="Path to point cloud .npy (N,5): x,y,z,grad,label")
    parser.add_argument("--y_idx", type=int, required=True, help="Slice index along the y-axis to visualize (0-based)")
    parser.add_argument("--ps_mask", type=float, default=2.0, help="Point size for mask points")
    parser.add_argument("--ps_pc",   type=float, default=2.5, help="Point size for PC points")
    args = parser.parse_args()

    # Load data
    mask_xzy = np.load(args.mask)     # expected shape (x,z,y)
    pc_xyzgl = np.load(args.pc)       # expected shape (N,5): (x,y,z,grad,label)

    if mask_xzy.ndim != 3:
        raise ValueError(f"Mask must be 3D (x,z,y). Got shape {mask_xzy.shape}")
    xdim, zdim, ydim = mask_xzy.shape
    if not (0 <= args.y_idx < ydim):
        raise IndexError(f"y_idx {args.y_idx} out of range [0, {ydim-1}] for mask shape {mask_xzy.shape}")

    # Filter point cloud rows that belong to this y slice
    pc_slice = pc_slice_from_full_pc(pc_xyzgl, y_idx=args.y_idx, round_to_voxel=True)

    # Visualize both together
    visualize_mask_and_pc_slice(
        mask_xzy=mask_xzy,
        pc_slice_xyzgl=pc_slice,
        y_idx=args.y_idx,
        point_size_mask=args.ps_mask,
        point_size_pc=args.ps_pc,
    )

if __name__ == "__main__":
    main()
