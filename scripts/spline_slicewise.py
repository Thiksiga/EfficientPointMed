import numpy as np

data = np.load(r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_20_kn5.npy")
print("Loaded shape:", data.shape, "| dtype:", data.dtype)
pts = data[data[:, 4] == 1, :3]
# xyz = data[:, :3]
print("pcd shape:", pts.shape, "| dtype:", pts.dtype)

sliced = pts[pts[:, 2] == 115]
print("sliced shape:", sliced.shape, "| dtype:", sliced.dtype)
print(sliced)

x, y = sliced[:, 0], sliced[:, 1]

import numpy as np
import matplotlib.pyplot as plt

# pts: (N,2)
pts = np.column_stack([x, y]).astype(float)

# ---- alpha-shape (Delaunay-based) ----
from scipy.spatial import Delaunay

def alpha_shape(points, alpha):
    tri = Delaunay(points)
    triangles = points[tri.simplices]                 # (M,3,2)
    a = np.linalg.norm(triangles[:,1]-triangles[:,0], axis=1)
    b = np.linalg.norm(triangles[:,2]-triangles[:,1], axis=1)
    c = np.linalg.norm(triangles[:,0]-triangles[:,2], axis=1)
    s = (a+b+c)/2.0
    area = np.sqrt(np.maximum(s*(s-a)*(s-b)*(s-c), 0.0))
    R = (a*b*c) / (4.0*area + 1e-12)                  # circumradius
    keep = R < (1.0/alpha)                            # alpha filter

    # build edges used by kept triangles
    edges = set()
    def add_edge(i,j):
        if i>j: i,j=j,i
        edges.add((i,j))
    for t, ok in zip(tri.simplices, keep):
        if not ok: continue
        add_edge(t[0], t[1]); add_edge(t[1], t[2]); add_edge(t[2], t[0])

    # order boundary edges into a polygon (simple chaining)
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for i,j in edges:
        adj[i].append(j); adj[j].append(i)
    # find a boundary loop
    start = next(iter(adj))
    poly_idx = [start]; prev = None; cur = start
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if prev is None or nbrs[0]!=prev else (nbrs[1] if len(nbrs)>1 else None)
        if nxt is None or nxt==start: break
        poly_idx.append(nxt); prev,cur = cur,nxt
    poly = points[poly_idx]
    return poly

# pick alpha (smaller => tighter). A good heuristic: alpha ~ 1 / median edge length
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
dists,_ = nbrs.kneighbors(pts)
med = np.median(dists[:,1])
alpha = 1.0/med

poly = alpha_shape(pts, alpha)

# polygon area (shoelace)
def poly_area(p):
    x,y = p[:,0], p[:,1]
    return 0.5*np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

area_alpha = poly_area(poly)

plt.figure(figsize=(6,6))
plt.scatter(pts[:,0], pts[:,1], s=5, alpha=0.35)
plt.fill(poly[:,0], poly[:,1], alpha=0.35, color='tab:cyan', label=f'Alpha area = {area_alpha:.2f}')
plt.plot(np.r_[poly[:,0], poly[0,0]], np.r_[poly[:,1], poly[0,1]], 'k-', lw=1)
plt.axis('equal'); plt.legend(); plt.title(f'Alpha shape (alpha≈{alpha:.3f})')
plt.show()


# import numpy as np
# from scipy.interpolate import CubicSpline
# import matplotlib.pyplot as plt
#
# x, y = sliced[:, 0], sliced[:, 1]
# cs = CubicSpline(x, y)
# xs = np.arange(-0.5, 9.6, 0.1)
# fig, ax = plt.subplots(figsize=(6.5, 4))
# ax.plot(x, y, 'o', label='data')
# ax.plot(xs, np.sin(xs), label='true')
# ax.plot(xs, cs(xs), label="S")
# ax.plot(xs, cs(xs, 1), label="S'")
# ax.plot(xs, cs(xs, 2), label="S''")
# ax.plot(xs, cs(xs, 3), label="S'''")
# ax.set_xlim(-0.5, 9.5)
# ax.legend(loc='lower left', ncol=2)
# plt.show()