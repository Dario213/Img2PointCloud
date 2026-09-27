import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

DATASETS = ["Box", "Entrance", "Statue", "Fountain"]

parser = argparse.ArgumentParser(description="Visualize dense point cloud for a dataset.")
parser.add_argument("dataset", choices=DATASETS, help="Dataset name")
parser.add_argument("--sparse", action="store_true", help="Load sparse points instead of dense")
args = parser.parse_args()

name = args.dataset.lower()
suffix = "sparse" if args.sparse else "dense"
ply_path = Path(f"output/{name}_{suffix}_points.ply")
txt_path = Path(f"output/{name}_{suffix}_points.txt")

if ply_path.exists():
    path = ply_path
elif txt_path.exists():
    path = txt_path
else:
    print(f"No point cloud found for {args.dataset} ({suffix}). Tried:")
    print(f"  {ply_path}")
    print(f"  {txt_path}")
    sys.exit(1)

print(f"Loading {path} ...")
pcd = o3d.io.read_point_cloud(str(path))
points = np.asarray(pcd.points)
print(f"Points: {len(points):,}")
if pcd.has_colors():
    print("Colors: yes")

o3d.visualization.draw_geometries(
    [pcd],
    window_name=f"{args.dataset} — {suffix} point cloud",
    width=1280,
    height=720,
)
