"""
Run the reconstruction pipeline on an arbitrary image directory.
Called by the web server for user-uploaded images.

Usage:
  python pipeline_custom.py <image_dir> <output_prefix> [--cameras|--no-cameras]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from submission import (
    parse_input_file,
    load_images,
    reconstruct_dense,
    run_colmap,
    reconstruct_dense_colmap,
    outlier_filter,
    remove_small_clusters,
    export_to_ply,
    save_submission_text,
    export_to_xyz,
    OUT_DIR,
)


def process_custom(image_dir: Path, out_prefix: str, has_cameras: bool):
    image_dir = Path(image_dir)
    print(f"[custom] image_dir={image_dir}  prefix={out_prefix}  known_poses={has_cameras}")

    imgs = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not imgs:
        print("[custom] ERROR: no images found")
        sys.exit(1)
    if len(imgs) < 2:
        print(f"[custom] ERROR: need at least 2 images for reconstruction (got {len(imgs)})")
        sys.exit(1)
    print(f"[custom] found {len(imgs)} image(s)")

    pts_all = col_all = None

    if has_cameras:
        cam_file = image_dir / "cameras.txt"
        if not cam_file.exists():
            print("[custom] ERROR: cameras.txt not found")
            sys.exit(1)
        cameras = parse_input_file(cam_file)
        cfg = {"images": image_dir, "poses_file": cam_file, "known_poses": True}
        images_bgr = load_images(cfg)
        print("[custom] running dense reconstruction with known poses …")
        from submission import build_K_from_fov
        import cv2
        h, w = cv2.imread(str(imgs[0])).shape[:2]
        K = build_K_from_fov(w, h)
        pts_all, col_all = reconstruct_dense(images_bgr, cameras, K)
    else:
        print("[custom] running COLMAP SfM …")
        colmap_out = image_dir.parent / f"{image_dir.name}_colmap"
        colmap_out.mkdir(exist_ok=True)
        result = run_colmap(image_dir, colmap_out)
        if result is None:
            print("[custom] ERROR: COLMAP failed")
            sys.exit(1)
        pts_sparse, col_sparse, model_dir = result
        pts_all, col_all = pts_sparse, col_sparse
        print(f"[custom] sparse: {len(pts_sparse)} pts — running dense …")
        pts_d, col_d = reconstruct_dense_colmap(image_dir, model_dir, max_pairs=4)
        if pts_d is not None and len(pts_d) > 0:
            pts_all = np.vstack([pts_all, pts_d])
            col_all = np.vstack([col_all, col_d])

    if pts_all is None or len(pts_all) == 0:
        print("[custom] ERROR: no points produced")
        sys.exit(1)

    pts_all, col_all = outlier_filter(pts_all, col_all, iqr_scale=3.0)
    pts_all, col_all = remove_small_clusters(pts_all, col_all)
    print(f"[custom] final: {len(pts_all)} pts")

    OUT_DIR.mkdir(exist_ok=True)
    export_to_ply(pts_all, col_all, OUT_DIR / f"{out_prefix}_dense_points.ply")
    save_submission_text(pts_all, OUT_DIR / f"{out_prefix}_dense_points.txt")
    export_to_xyz(pts_all, col_all, OUT_DIR / f"{out_prefix}_dense_points.xyz")
    print(f"[custom] saved to {OUT_DIR}/{out_prefix}_dense_points.*")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image_dir")
    p.add_argument("out_prefix")
    p.add_argument("--cameras", dest="has_cameras", action="store_true")
    p.add_argument("--no-cameras", dest="has_cameras", action="store_false")
    p.set_defaults(has_cameras=False)
    args = p.parse_args()
    process_custom(args.image_dir, args.out_prefix, args.has_cameras)
