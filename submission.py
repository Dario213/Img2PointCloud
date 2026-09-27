import argparse
from pathlib import Path

import pycolmap

import cv2
import numpy as np
import scipy.spatial.transform as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE     = Path(__file__).parent
IMG_ROOT = HERE / "StemGames2026_ProjectTask/TestImages"
OUT_DIR  = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

RATIO_THRESHOLD = 0.75
MIN_MATCHES     = 35
FLANN_TREES     = 15
FLANN_CHECKS    = 250
SIFT_NFEATURES  = 10000
RANSAC_PIX      = 1.5

SCALE          = 1
NUM_DISP       = 256   # multiple of 16; wider for up-to-45° pairs
BLOCK_SIZE     = 11
TEXTURE_THRESH = 15.0  # lower → more surface coverage; higher → fewer textureless-region errors
DOT_THRESH     = 0.70  # pairs within ~45°; more scene coverage

def parse_input_file(path):
    cameras, cam = [], {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()

        def _vals(line):
            return [float(x.split("=")[1]) for x in line.split()[1:] if "=" in x]

        if line.startswith("CamPosition:"):
            cam["pos"] = tuple(_vals(line))
        elif line.startswith("CamForward:"):
            cam["forward"] = tuple(_vals(line))
        elif line.startswith("CamRight"):
            cam["right"] = tuple(_vals(line))
        elif line.startswith("CamUp"):
            cam["up"] = tuple(_vals(line))
            if all(k in cam for k in ("pos", "forward", "right", "up")):
                cameras.append(cam)
                cam = {}
    return cameras


def parse_k_txt(path):
    text = Path(path).read_text()
    nums = [float(x)
            for x in text.replace("[", " ").replace("]", " ")
                         .replace(";", " ").split()
            if x not in ("K", "=")]
    return np.array(nums, dtype=np.float64).reshape(3, 3)


def build_extrinsics(cam):
    fwd = np.array(cam["forward"], float); fwd /= np.linalg.norm(fwd) + 1e-12
    rgt = np.array(cam["right"],   float); rgt /= np.linalg.norm(rgt) + 1e-12
    up  = np.array(cam["up"],      float); up  /= np.linalg.norm(up)  + 1e-12
    pos = np.array(cam["pos"],     float)
    R = np.stack([rgt, -up, fwd], axis=0)
    t = -R @ pos
    return R, t


def rot_to_quat(R):
    """Convert rotation matrix to quaternion [qw, qx, qy, qz].
    Handles det=-1 (improper rotation from Y-up↔Y-down mapping) by flipping row 2.
    """
    R = np.array(R, dtype=float)
    if np.linalg.det(R) < 0:
        R[2] = -R[2]
    q = st.Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return q[3], q[0], q[1], q[2]             # → qw, qx, qy, qz


def build_K_from_fov(w, h, fov_deg=90.0):
    f = w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0,      1.0]], dtype=np.float64)


def get_line_equation(pixel_row, pixel_col, res_x, res_y,
                      cam_forward, cam_right, cam_up, cam_position):
    coeff_right = 2.0 * (pixel_col - res_x / 2.0 + 0.5) / res_x
    coeff_up    = -2.0 * (pixel_row - res_y / 2.0 + 0.5) / res_y
    coeff_up   *= res_y / res_x
    fwd = np.asarray(cam_forward, float)
    rgt = np.asarray(cam_right,   float)
    up  = np.asarray(cam_up,      float)
    direction = fwd + coeff_right * rgt + coeff_up * up
    return (np.asarray(cam_position, float),
            direction / (np.linalg.norm(direction) + 1e-12))


def calculate_3d_point(origin1, dir1, origin2, dir2):
    o1, d1 = np.asarray(origin1, float), np.asarray(dir1, float)
    o2, d2 = np.asarray(origin2, float), np.asarray(dir2, float)
    n  = np.cross(d1, d2)
    n1 = np.cross(d1, n)
    n2 = np.cross(d2, n)
    denom1 = np.dot(d1, n2)
    denom2 = np.dot(d2, n1)
    if abs(denom1) < 1e-8 or abs(denom2) < 1e-8:
        return None
    c1 = o1 + d1 * (np.dot(o2 - o1, n2) / denom1)
    c2 = o2 + d2 * (np.dot(o1 - o2, n1) / denom2)
    return (c1 + c2) / 2.0


def _reproject(pt3d, cam, res_x, res_y):
    """
    Inverse of GetLineEquation: world pt3d → (col, row) pixel using the
    competition's camera model (90° FOV, no distortion).
    Returns None if the point is behind the camera.
    """
    v   = pt3d - np.asarray(cam["pos"], float)
    fwd = np.asarray(cam["forward"], float)
    rgt = np.asarray(cam["right"],   float)
    up  = np.asarray(cam["up"],      float)
    vf  = np.dot(v, fwd)
    if vf <= 0:
        return None
    coeff_right = np.dot(v, rgt) / vf
    coeff_up    = np.dot(v, up)  / vf
    col = coeff_right * res_x / 2.0 + res_x / 2.0 - 0.5
    row = -coeff_up   * res_x / 2.0 + res_y / 2.0 - 0.5
    return col, row


def overlap_pairs(n, cameras, step=2, min_dot=0.3):
    raw = list(dict.fromkeys(
        [(i, (i + 1) % n) for i in range(n)] +
        [(i, (i + step) % n) for i in range(n)]
    ))
    kept = []
    for i, j in raw:
        fi = np.array(cameras[i]["forward"], float)
        fj = np.array(cameras[j]["forward"], float)
        if np.dot(fi / np.linalg.norm(fi), fj / np.linalg.norm(fj)) > min_dot:
            kept.append((i, j))
    return kept


def sift_detect(sift, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def flann_match(des_a, des_b):
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=FLANN_TREES),
        dict(checks=FLANN_CHECKS),
    )
    raw = flann.knnMatch(des_a, des_b, k=2)
    return [m for pair in raw if len(pair) == 2
            for m, n in (pair,)
            if m.distance < RATIO_THRESHOLD * n.distance]


def reconstruct_sparse(imgs, cameras, res_x, res_y):
    print("  [Sparse] SIFT detection …")
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    kps_descs = [sift_detect(sift, img) for img in imgs]

    n = len(imgs)
    pairs = overlap_pairs(n, cameras, step=2)
    print(f"  [Sparse] {len(pairs)} pairs after overlap filter")

    all_pts, all_col = [], []
    matched = 0
    for i, j in pairs:
        kp_a, des_a = kps_descs[i]
        kp_b, des_b = kps_descs[j]
        if des_a is None or des_b is None:
            continue
        good = flann_match(des_a, des_b)
        if len(good) < MIN_MATCHES:
            continue

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
        _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, RANSAC_PIX, 0.999)
        if mask is None:
            continue
        good = [m for m, k in zip(good, mask.ravel()) if k]
        if len(good) < MIN_MATCHES:
            continue
        matched += 1

        cam_a, cam_b = cameras[i], cameras[j]
        for m in good:
            pa = kp_a[m.queryIdx].pt
            pb = kp_b[m.trainIdx].pt

            o_a, d_a = get_line_equation(pa[1], pa[0], res_x, res_y,
                                         cam_a["forward"], cam_a["right"],
                                         cam_a["up"],      cam_a["pos"])
            o_b, d_b = get_line_equation(pb[1], pb[0], res_x, res_y,
                                         cam_b["forward"], cam_b["right"],
                                         cam_b["up"],      cam_b["pos"])

            pt3d = calculate_3d_point(o_a, d_a, o_b, d_b)
            if pt3d is None:
                continue

            if np.dot(pt3d - o_a, d_a) < 0 or np.dot(pt3d - o_b, d_b) < 0:
                continue
            dist_a = np.linalg.norm(pt3d - o_a)
            if dist_a > 3000:
                continue

            rp_a = _reproject(pt3d, cam_a, res_x, res_y)
            rp_b = _reproject(pt3d, cam_b, res_x, res_y)
            if rp_a is None or rp_b is None:
                continue
            if max(np.hypot(rp_a[0] - pa[0], rp_a[1] - pa[1]),
                   np.hypot(rp_b[0] - pb[0], rp_b[1] - pb[1])) > 2.0:
                continue

            row = int(np.clip(pa[1], 0, imgs[i].shape[0] - 1))
            col = int(np.clip(pa[0], 0, imgs[i].shape[1] - 1))
            bgr = imgs[i][row, col]
            all_pts.append(pt3d)
            all_col.append(bgr[::-1].astype(float) / 255.0)

    print(f"  [Sparse] {matched} pairs matched → {len(all_pts)} raw points")
    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3))

    pts, col = outlier_filter(np.array(all_pts), np.array(all_col))
    print(f"  [Sparse] {len(pts):,} points after outlier filter")
    return pts, col


def _feat_opts(n_features=SIFT_NFEATURES):
    opts = pycolmap.FeatureExtractionOptions()
    opts.sift.max_num_features = n_features
    return opts


def _match_opts(ransac_px=RANSAC_PIX):
    m = pycolmap.FeatureMatchingOptions()
    m.guided_matching = True
    v = pycolmap.TwoViewGeometryOptions()
    v.ransac.max_error = ransac_px
    return m, v


def _pts_from_recon(recon):
    pts, cols = [], []
    for pt in recon.points3D.values():
        pts.append(pt.xyz)
        cols.append(pt.color.astype(float) / 255.0)
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float32)


def run_sparse_colmap_known(img_dir, cameras, K, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    db     = out_dir / "colmap.db"
    sparse = out_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = int(round(cx * 2)), int(round(cy * 2))

    print("  [COLMAP] feature extraction …")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model  = "PINHOLE"
    reader.camera_params = f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}"
    pycolmap.extract_features(
        db, img_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=_feat_opts(),
        device=pycolmap.Device.cpu,
    )

    print("  [COLMAP] exhaustive matching …")
    m_opts, v_opts = _match_opts()
    pycolmap.match_exhaustive(db, matching_options=m_opts, verification_options=v_opts)

    # Query actual image IDs assigned by the database (filesystem order ≠ our numbering)
    import sqlite3
    conn = sqlite3.connect(str(db))
    name_to_db_id = {name: iid for iid, name in conn.execute("SELECT image_id, name FROM images")}
    conn.close()

    # Write known poses using the exact IDs from the database
    prefix = img_dir.name.lower()
    with open(sparse / "cameras.txt", "w") as f:
        f.write("# ID Model W H params\n")
        f.write(f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    with open(sparse / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for i, cam in enumerate(cameras):
            img_name = f"{prefix}{i+1}.png"
            img_id = name_to_db_id.get(img_name)
            if img_id is None:
                print(f"  WARNING: {img_name} not in database, skipping")
                continue
            R, t = build_extrinsics(cam)
            qw, qx, qy, qz = rot_to_quat(R)
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {img_name}\n\n")
    (sparse / "points3D.txt").write_text("# 3D point list\n")

    recon = pycolmap.Reconstruction()
    recon.read_text(str(sparse))

    tri_opts = pycolmap.IncrementalPipelineOptions()
    tri_opts.triangulation.min_angle                  = 1.0
    tri_opts.triangulation.complete_max_reproj_error  = 4.0
    tri_opts.triangulation.merge_max_reproj_error     = 4.0

    print("  [COLMAP] triangulating …")
    recon_out = pycolmap.triangulate_points(
        recon, db, img_dir, str(sparse), options=tri_opts
    )

    if recon_out is None or recon_out.num_points3D() == 0:
        print("  [COLMAP] No points triangulated.")
        return None

    pts, col = _pts_from_recon(recon_out)
    print(f"  [COLMAP] {len(pts):,} triangulated points")
    return pts, col


def dense_pair(img1, img2, K_full, R1, t1, R2, t2):
    h0, w0 = img1.shape[:2]
    w, h = int(w0 * SCALE), int(h0 * SCALE)
    i1, i2 = cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))

    Ks = K_full.copy()
    Ks[0] *= SCALE; Ks[1] *= SCALE
    dist = np.zeros(5)

    R_rel = R2 @ R1.T
    t_rel = (t2 - R_rel @ t1).reshape(3, 1)
    if np.linalg.norm(t_rel) < 1e-6:
        return np.empty((0, 3)), np.empty((0, 3))

    R1r, R2r, P1r, P2r, Q, roi1, roi2 = cv2.stereoRectify(
        Ks, dist, Ks, dist, (w, h), R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1,
    )
    m1x, m1y = cv2.initUndistortRectifyMap(Ks, dist, R1r, P1r, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(Ks, dist, R2r, P2r, (w, h), cv2.CV_32FC1)
    r1 = cv2.remap(i1, m1x, m1y, cv2.INTER_LINEAR)
    r2 = cv2.remap(i2, m2x, m2y, cv2.INTER_LINEAR)

    # Restrict to the intersection of both valid ROIs so border padding is excluded
    roi_mask = np.zeros((h, w), dtype=bool)
    x1, y1, rw1, rh1 = roi1
    x2, y2, rw2, rh2 = roi2
    rx = max(x1, x2); ry = max(y1, y2)
    rw = min(x1 + rw1, x2 + rw2) - rx
    rh = min(y1 + rh1, y2 + rh2) - ry
    if rw > 0 and rh > 0:
        roi_mask[ry:ry + rh, rx:rx + rw] = True
    else:
        roi_mask[:] = True  # fallback: trust SGBM quality gates alone

    gray1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    tex_mask = grad > TEXTURE_THRESH
    tex_mask = cv2.dilate(tex_mask.astype(np.uint8),
                          np.ones((9, 9), np.uint8)).astype(bool)

    bs = BLOCK_SIZE
    sgbm = cv2.StereoSGBM_create(
        minDisparity=1, numDisparities=NUM_DISP, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,
        disp12MaxDiff=1, uniquenessRatio=20,
        speckleWindowSize=200, speckleRange=8,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
    disp_raw = sgbm.compute(g1, g2)
    try:
        wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        disp_raw = wls.filter(disp_raw, r1,
                              disparity_map_right=cv2.ximgproc.createRightMatcher(sgbm).compute(g2, g1))
    except AttributeError:
        pass  # opencv-contrib not available, use raw disparity
    disp = disp_raw.astype(np.float32) / 16.0

    pts_rect = cv2.reprojectImageTo3D(disp, Q)
    baseline = float(np.linalg.norm(t_rel))
    max_depth = baseline * 10         # allow deeper scene; disp>3 still blocks fog
    valid = (disp > 3.0) \
          & np.all(np.isfinite(pts_rect), axis=2) \
          & (pts_rect[:, :, 2] > 0) \
          & (pts_rect[:, :, 2] < max_depth) \
          & tex_mask \
          & roi_mask

    pts_r = pts_rect[valid]
    pts_w = (R1.T @ R1r.T @ pts_r.T).T + (-R1.T @ t1)
    colors = cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)[valid].astype(np.float32) / 255.0
    return pts_w, colors


def reconstruct_dense(imgs, cameras, K):
    print("  [Dense] Running StereoSGBM pairs …")
    Rts  = [build_extrinsics(c) for c in cameras]
    fwds = [np.array(c["forward"], float) for c in cameras]
    fwds = [f / (np.linalg.norm(f) + 1e-12) for f in fwds]
    n = len(imgs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.dot(fwds[i], fwds[j]) > DOT_THRESH]
    print(f"  [Dense] {len(pairs)} pairs after angle filter")

    all_pts, all_col = [], []
    for i, j in pairs:
        R1, t1 = Rts[i]; R2, t2 = Rts[j]
        print(f"    pair ({i+1:2d},{j+1:2d}) …", end=" ", flush=True)
        pts, col = dense_pair(imgs[i], imgs[j], K, R1, t1, R2, t2)
        print(f"{len(pts):,}")
        if len(pts):
            all_pts.append(pts); all_col.append(col)

    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3))

    pts_all = np.vstack(all_pts)
    col_all = np.vstack(all_col)
    pts_all, col_all = outlier_filter(pts_all, col_all, iqr_scale=3.0)
    print(f"  [Dense] {len(pts_all):,} points after outlier filter")
    pts_all, col_all = remove_small_clusters(pts_all, col_all)
    print(f"  [Dense] {len(pts_all):,} points after cluster filter")
    return pts_all, col_all


def run_colmap(image_dir, out_dir, K=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    db     = out_dir / "colmap.db"
    sparse = out_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    reader = pycolmap.ImageReaderOptions()
    if K is not None:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        reader.camera_model  = "PINHOLE"
        reader.camera_params = f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}"

    print("  [COLMAP] feature extraction …")
    pycolmap.extract_features(
        db, image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        device=pycolmap.Device.cpu,
    )

    print("  [COLMAP] exhaustive matching …")
    m_opts, v_opts = _match_opts()
    pycolmap.match_exhaustive(db, matching_options=m_opts, verification_options=v_opts)

    map_opts = pycolmap.IncrementalPipelineOptions()
    map_opts.min_num_matches                       = 10
    map_opts.ba_refine_focal_length                = (K is None)
    map_opts.ba_refine_extra_params                = False
    map_opts.mapper.init_min_num_inliers           = 15
    map_opts.mapper.abs_pose_min_num_inliers       = 10
    map_opts.mapper.abs_pose_min_inlier_ratio      = 0.20
    map_opts.mapper.filter_max_reproj_error        = 8.0
    map_opts.triangulation.min_angle               = 1.0
    map_opts.triangulation.complete_max_reproj_error = 8.0

    print("  [COLMAP] incremental mapping …")
    models = pycolmap.incremental_mapping(db, image_dir, str(sparse), options=map_opts)
    if not models:
        print("  [COLMAP] No reconstruction produced.")
        return None

    best = max(models.values(), key=lambda r: r.num_points3D())
    if len(models) > 1:
        print(f"  [COLMAP] Picked largest of {len(models)} models ({best.num_points3D()} pts)")

    # Densify with point_triangulator
    tri_opts = pycolmap.IncrementalPipelineOptions()
    tri_opts.triangulation.min_angle                  = 1.0
    tri_opts.triangulation.complete_max_reproj_error  = 8.0
    tri_opts.triangulation.merge_max_reproj_error     = 8.0
    dense_dir = sparse / "dense"
    dense_dir.mkdir(exist_ok=True)
    print("  [COLMAP] densifying …")
    recon_dense = pycolmap.triangulate_points(
        best, db, image_dir, str(dense_dir), options=tri_opts
    )
    final = recon_dense if (recon_dense and recon_dense.num_points3D() > best.num_points3D()) else best

    # Write text files for reconstruct_dense_colmap
    model_dir = dense_dir if final is recon_dense else sparse / "best"
    model_dir.mkdir(exist_ok=True)
    final.write_text(str(model_dir))

    pts, col = _pts_from_recon(final)
    if len(pts) == 0:
        print("  [COLMAP] No points extracted.")
        return None

    print(f"  [COLMAP] {len(pts):,} sparse 3D points")
    return pts, col, model_dir


def parse_colmap_points3d(path):
    path = Path(path)
    if not path.exists():
        return np.empty((0, 3)), np.empty((0, 3))

    pts, cols = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        pts.append([float(p[1]), float(p[2]), float(p[3])])
        cols.append([int(p[4]) / 255., int(p[5]) / 255., int(p[6]) / 255.])
    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float32)


def parse_colmap_cameras_txt(path):
    import scipy.spatial.transform as st
    cams = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        cid, model = int(p[0]), p[1]
        w, h = int(p[2]), int(p[3])
        params = list(map(float, p[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            fx = fy = params[0]; cx, cy = params[1], params[2]
        cams[cid] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]], dtype=np.float64)
    return cams


def parse_colmap_images_txt(path):
    import scipy.spatial.transform as st
    imgs = []
    lines = [l for l in Path(path).read_text().splitlines()
             if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        qw,qx,qy,qz = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        tx,ty,tz    = float(p[5]),float(p[6]),float(p[7])
        R = st.Rotation.from_quat([qx,qy,qz,qw]).as_matrix()
        t = np.array([tx,ty,tz])
        imgs.append({"name": p[9], "R": R, "t": t,
                     "cam_id": int(p[8]), "id": int(p[0])})
        i += 2
    return imgs


def reconstruct_dense_colmap(image_dir, model_dir, K_override=None, max_pairs=5):
    """
    Dense SGBM using COLMAP-estimated poses (for Statue / Fountain).
    K_override: pass the known K matrix from K.txt to avoid using COLMAP's
                possibly re-estimated (wrong) intrinsics.
    """
    cam_Ks_from_file = parse_colmap_cameras_txt(model_dir / "cameras.txt")
    # Use known K when available — COLMAP may refine it to wrong values
    cam_Ks = {cid: K_override if K_override is not None else K
              for cid, K in cam_Ks_from_file.items()}
    img_metas = parse_colmap_images_txt(model_dir / "images.txt")
    if not img_metas:
        print("  [Dense-COLMAP] No poses in images.txt"); return np.empty((0,3)), np.empty((0,3))

    # Load images by name
    loaded = {}
    for m in img_metas:
        p = image_dir / m["name"]
        img = cv2.imread(str(p))
        if img is not None:
            loaded[m["id"]] = img

    if not loaded:
        print("  [Dense-COLMAP] No images loaded"); return np.empty((0,3)), np.empty((0,3))

    def _cam_center(R, t): return -R.T @ t

    def _angle(m1, m2):
        cos_a = np.clip(np.dot(m1["R"][2], m2["R"][2]), -1, 1)
        return np.degrees(np.arccos(cos_a))

    all_pts, all_col = [], []
    print(f"  [Dense-COLMAP] {len(img_metas)} views, {max_pairs} neighbors each …")

    for ref in img_metas:
        if ref["id"] not in loaded: continue
        img_ref = loaded[ref["id"]]
        K = cam_Ks[ref["cam_id"]]
        R1, t1 = ref["R"], ref["t"]

        neighbors = sorted(
            [m for m in img_metas if m["id"] != ref["id"] and m["id"] in loaded],
            key=lambda m: _angle(ref, m)
        )[:max_pairs]

        for nbr in neighbors:
            print(f"    {ref['name']} ↔ {nbr['name']} …", end=" ", flush=True)
            pts, col = dense_pair(img_ref, loaded[nbr["id"]], K,
                                  R1, t1, nbr["R"], nbr["t"])
            print(f"{len(pts):,}")
            if len(pts): all_pts.append(pts); all_col.append(col)

    if not all_pts:
        return np.empty((0,3)), np.empty((0,3))
    pts_all = np.vstack(all_pts)
    col_all = np.vstack(all_col)
    pts_all, col_all = outlier_filter(pts_all, col_all)
    pts_all, col_all = remove_small_clusters(pts_all, col_all)
    print(f"  [Dense-COLMAP] {len(pts_all):,} points after filter")
    return pts_all, col_all


def filter_dense_by_sparse(pts_d, col_d, pts_s, bbox_expand=4, prox_factor=1):
    if len(pts_s) < 10 or len(pts_d) == 0:
        return pts_d, col_d

    lo = np.percentile(pts_s, 10, axis=0)
    hi = np.percentile(pts_s, 90, axis=0)
    center, half = (lo + hi) / 2, (hi - lo) / 2 * bbox_expand
    lo, hi = center - half, center + half
    bbox_mask = np.all((pts_d >= lo) & (pts_d <= hi), axis=1)
    pts_d, col_d = pts_d[bbox_mask], col_d[bbox_mask]
    if len(pts_d) == 0:
        return pts_d, col_d

    bbox_diag = float(np.linalg.norm(hi - lo))
    radius = bbox_diag * prox_factor
    keep = np.zeros(len(pts_d), dtype=bool)
    chunk = 20_000
    for s in range(0, len(pts_d), chunk):
        e = min(s + chunk, len(pts_d))
        d = np.linalg.norm(pts_d[s:e, None, :] - pts_s[None, :, :], axis=-1)
        keep[s:e] = d.min(axis=1) < radius
    return pts_d[keep], col_d[keep]


def outlier_filter(pts, colors, iqr_scale=1.5):
    if len(pts) < 10:
        return pts, colors
    q1 = np.percentile(pts, 25, axis=0)
    q3 = np.percentile(pts, 75, axis=0)
    iqr = q3 - q1
    lo  = q1 - iqr_scale * iqr
    hi  = q3 + iqr_scale * iqr
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[mask], colors[mask]


def remove_small_clusters(pts, colors, radius_frac=0.04, min_frac=0.001):
    """
    Remove isolated point clusters via connected components on a subsampled graph.
    Clusters with fewer than min_frac * total_points are discarded.
    radius_frac: neighbor radius as fraction of the scene bounding-box diagonal.
    """
    if len(pts) < 200:
        return pts, colors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(pts)
    radius = float(np.linalg.norm(pts.max(0) - pts.min(0))) * radius_frac
    min_pts = max(int(n * min_frac), 50)

    # Subsample for a manageable graph while preserving cluster structure
    MAX = 100_000
    if n > MAX:
        sub = np.random.choice(n, MAX, replace=False)
        pts_sub = pts[sub]
    else:
        sub = np.arange(n)
        pts_sub = pts

    m = len(pts_sub)
    pairs = cKDTree(pts_sub).query_pairs(r=radius, output_type='ndarray')

    if len(pairs):
        r_arr = np.concatenate([pairs[:, 0], pairs[:, 1]])
        c_arr = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = csr_matrix((np.ones(len(r_arr)), (r_arr, c_arr)), shape=(m, m))
    else:
        graph = csr_matrix((m, m))

    _, labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(labels)
    min_sub = max(int(m * min_frac), 5)
    large = np.where(comp_sizes >= min_sub)[0]

    # Project every original point to its nearest subsampled point's label
    _, nn = cKDTree(pts_sub).query(pts, k=1)
    pt_labels = labels[nn]
    mask = np.isin(pt_labels, large)

    removed = int((~mask).sum())
    if removed:
        print(f"  [Cluster] removed {removed:,} pts ({removed/n*100:.1f}%) in"
              f" {len(labels) - len(large)} small clusters")
    return pts[mask], colors[mask]


def save_submission_text(pts, path):
    np.savetxt(str(path), pts, fmt="%f %f %f")
    print(f"  Submission .txt       → {path}  ({len(pts):,} pts)")


def export_to_ply(pts, colors, path):
    c8 = (np.clip(np.asarray(colors, float), 0, 1) * 255).astype(np.uint8)
    with open(str(path), "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for (x, y, z), (r, g, b) in zip(pts, c8):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"  PLY (MeshLab)         → {path}  ({len(pts):,} pts)")


def export_to_xyz(pts, colors, path):
    c8 = (np.clip(np.asarray(colors, float), 0, 1) * 255).astype(np.uint8)
    with open(str(path), "w") as f:
        for (x, y, z), (r, g, b) in zip(pts, c8):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"  XYZ (CloudCompare)    → {path}  ({len(pts):,} pts)")


def visualize(pts, colors, cam_positions=None, title="Point Cloud", save_path=None):
    MAX_VIS = 200_000
    if len(pts) > MAX_VIS:
        idx   = np.random.choice(len(pts), MAX_VIS, replace=False)
        pts_v = pts[idx]; col_v = colors[idx]
    else:
        pts_v = pts; col_v = colors

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(title, fontsize=14)

    ax3 = fig.add_subplot(131, projection="3d")
    ax3.scatter(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2],
                c=col_v, s=0.3, linewidths=0)
    if cam_positions is not None:
        cp = np.array(cam_positions)
        ax3.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
                    c="red", s=60, marker="^", zorder=5, label="cameras")
        ax3.legend(loc="upper right", fontsize=7)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("3D view")

    def _lim(arr, pad=0.1):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        m = (hi - lo) * pad
        return lo - m, hi + m

    ax_xy = fig.add_subplot(132)
    ax_xy.scatter(pts_v[:, 0], pts_v[:, 1], c=col_v, s=0.3, linewidths=0)
    if cam_positions is not None:
        ax_xy.scatter(cp[:, 0], cp[:, 1], c="red", s=40, marker="^", zorder=5)
    ax_xy.set_xlim(*_lim(pts_v[:, 0])); ax_xy.set_ylim(*_lim(pts_v[:, 1]))
    ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y"); ax_xy.set_title("Top view (XY)")

    ax_xz = fig.add_subplot(133)
    ax_xz.scatter(pts_v[:, 0], pts_v[:, 2], c=col_v, s=0.3, linewidths=0)
    if cam_positions is not None:
        ax_xz.scatter(cp[:, 0], cp[:, 2], c="red", s=40, marker="^", zorder=5)
    ax_xz.set_xlim(*_lim(pts_v[:, 0])); ax_xz.set_ylim(*_lim(pts_v[:, 2]))
    ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z"); ax_xz.set_title("Side view (XZ)")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
        print(f"  Visualization         → {save_path}")
    plt.close()


def make_datasets():
    return {
        "Box": {
            "images":  IMG_ROOT / "Box",
            "prefix":  "box",      "ext": ".png",  "n": 12,
            "cameras": parse_input_file(IMG_ROOT / "Box/boxInput.txt"),
            "res":     (1920, 1080), "fov": 90.0, "mode": "known",
        },
        "Entrance": {
            "images":  IMG_ROOT / "Entrance",
            "prefix":  "entrance", "ext": ".png",  "n": 12,
            "cameras": parse_input_file(IMG_ROOT / "Entrance/entranceInput.txt"),
            "res":     (1920, 1080), "fov": 90.0, "mode": "known",
        },
        "Statue": {
            "images":  IMG_ROOT / "Statue",
            "prefix":  "statue",   "ext": ".png",  "n": 18,
            "K":       parse_k_txt(IMG_ROOT / "Statue/K.txt"),
            "mode":    "unknown",
        },
        "Fountain": {
            "images":  IMG_ROOT / "Fountain",
            "prefix":  "fountain", "ext": ".jpg",  "n": 11,
            "K":       parse_k_txt(IMG_ROOT / "Fountain/K.txt"),
            "mode":    "unknown",
        },
    }


def load_images(cfg):
    imgs = []
    for i in range(1, cfg["n"] + 1):
        p = cfg["images"] / f"{cfg['prefix']}{i}{cfg['ext']}"
        img = cv2.imread(str(p))
        if img is None:
            print(f"  WARNING: cannot read {p}")
            continue
        imgs.append(img)
    return imgs


def process_dataset(name, cfg, do_sparse=True, do_dense=True):
    print(f"\n{'='*60}")
    print(f"Dataset : {name}  |  mode = {cfg['mode']}")
    print(f"{'='*60}")

    imgs = load_images(cfg)
    if not imgs:
        print("  ERROR: no images loaded"); return
    h, w = imgs[0].shape[:2]
    prefix = cfg["prefix"]

    if cfg["mode"] == "known":
        cameras = cfg["cameras"]
        K       = build_K_from_fov(w, h, cfg["fov"])
        res_x, res_y = cfg["res"]
        cam_positions = [np.array(c["pos"]) for c in cameras]
        print(f"  Loaded {len(imgs)} images  {w}×{h}")

        sparse_pts = None
        if do_sparse:
            print("\n  ── Approach A: Sparse (COLMAP point_triangulator) ──")
            colmap_out = OUT_DIR / f"colmap_{prefix}"
            colmap_out.mkdir(exist_ok=True)
            result_s = run_sparse_colmap_known(cfg["images"], cameras, K, colmap_out)
            if result_s is not None:
                pts_s, col_s = result_s
                pts_s, col_s = outlier_filter(pts_s, col_s)
                if len(pts_s):
                    sparse_pts = pts_s
                    save_submission_text(pts_s, OUT_DIR / f"{prefix}_sparse_points.txt")
                    export_to_ply(pts_s, col_s, OUT_DIR / f"{prefix}_sparse_points.ply")
                    export_to_xyz(pts_s, col_s, OUT_DIR / f"{prefix}_sparse_points.xyz")
                    visualize(pts_s, col_s, cam_positions,
                              title=f"{name} – Sparse",
                              save_path=OUT_DIR / f"{prefix}_sparse_vis.png")

        if do_dense:
            print("\n  ── Approach B: Dense (StereoSGBM) ──")
            pts_d, col_d = reconstruct_dense(imgs, cameras, K)
            if len(pts_d) and sparse_pts is not None:
                pts_d, col_d = filter_dense_by_sparse(pts_d, col_d, sparse_pts,
                                                       bbox_expand=2, prox_factor=2)
                print(f"  [Dense] {len(pts_d):,} points after sparse-guided filter")
            if len(pts_d):
                save_submission_text(pts_d, OUT_DIR / f"{prefix}_dense_points.txt")
                export_to_ply(pts_d, col_d, OUT_DIR / f"{prefix}_dense_points.ply")
                export_to_xyz(pts_d, col_d, OUT_DIR / f"{prefix}_dense_points.xyz")
                visualize(pts_d, col_d, cam_positions,
                          title=f"{name} – Dense",
                          save_path=OUT_DIR / f"{prefix}_dense_vis.png")

    else:
        K = cfg["K"]
        print(f"  Loaded {len(imgs)} images  {w}×{h}")
        print(f"  K (from K.txt):\n{K}")
        print("\n  ── Approach C: COLMAP SfM + Dense SGBM ──")
        colmap_out = OUT_DIR / f"colmap_{prefix}"
        colmap_out.mkdir(exist_ok=True)
        result = run_colmap(cfg["images"], colmap_out, K)
        if result is not None:
            pts_sparse, col_sparse, model_dir = result
            pts_sparse, col_sparse = outlier_filter(pts_sparse, col_sparse)

            # Dense SGBM using COLMAP-estimated poses
            if do_dense:
                print("\n  ── Dense SGBM from COLMAP poses ──")
                pts_d, col_d = reconstruct_dense_colmap(cfg["images"], model_dir, K_override=K)
                if len(pts_d):
                    save_submission_text(pts_d, OUT_DIR / f"{prefix}_dense_points.txt")
                    export_to_ply(pts_d, col_d, OUT_DIR / f"{prefix}_dense_points.ply")
                    export_to_xyz(pts_d, col_d, OUT_DIR / f"{prefix}_dense_points.xyz")
                    visualize(pts_d, col_d,
                              title=f"{name} – Dense (COLMAP poses)",
                              save_path=OUT_DIR / f"{prefix}_dense_vis.png")

            # Also save the sparse cloud
            if len(pts_sparse):
                save_submission_text(pts_sparse, OUT_DIR / f"{prefix}_sparse_points.txt")
                export_to_ply(pts_sparse, col_sparse, OUT_DIR / f"{prefix}_sparse_points.ply")
                export_to_xyz(pts_sparse, col_sparse, OUT_DIR / f"{prefix}_sparse_points.xyz")
                visualize(pts_sparse, col_sparse,
                          title=f"{name} – COLMAP sparse",
                          save_path=OUT_DIR / f"{prefix}_colmap_vis.png")


def main():
    parser = argparse.ArgumentParser(
        description="STEM Games 2026 — 3D Point Cloud Reconstruction"
    )
    parser.add_argument(
        "dataset", nargs="?", default="all",
        choices=["Box", "Entrance", "Statue", "Fountain", "all"],
        help="Dataset to process (default: all)",
    )
    parser.add_argument("--sparse", action="store_true", help="Sparse only (Approach A)")
    parser.add_argument("--dense",  action="store_true", help="Dense only (Approach B)")
    args = parser.parse_args()

    do_sparse = not args.dense
    do_dense  = not args.sparse

    datasets = make_datasets()
    if args.dataset == "all":
        for name, cfg in datasets.items():
            process_dataset(name, cfg, do_sparse, do_dense)
    else:
        process_dataset(args.dataset, datasets[args.dataset], do_sparse, do_dense)

if __name__ == "__main__":
    main()