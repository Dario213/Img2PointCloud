# Img2PointCloud

**Task:** Reconstruct 3D point clouds from multi-view images across four datasets.

**Final solution: `submission.py` and web viewer** 

---

## Datasets

| Dataset  | Images | Camera poses | Notes |
|----------|--------|--------------|-------|
| Box      | 12 PNG | Known (`boxInput.txt`) | Wooden crate in sandy outdoor scene, 90° FOV, 1920×1080 |
| Entrance | 12 PNG | Known (`entranceInput.txt`) | Arched entrance with pillars, 90° FOV, 1920×1080 |
| Statue   | 18 PNG | Unknown | Statue figure, intrinsics in `K.txt` |
| Fountain | 11 JPG | Unknown | Outdoor fountain, intrinsics in `K.txt` |

For Box and Entrance, each `*Input.txt` file provides per-image camera position, forward vector, right vector, and up vector. For Statue and Fountain only the intrinsic matrix K is given — camera poses must be estimated from the images themselves.

---

## Final Pipeline

### Known poses — Box, Entrance

**Sparse (COLMAP point triangulator)**

1. Extract SIFT features with `pycolmap.extract_features` (10,000 features per image).
2. Run exhaustive feature matching (`pycolmap.match_exhaustive`) with guided matching.
3. Convert known poses to COLMAP format:
   - Build rotation matrix `R = [right, -up, forward]` and translation `t = -R @ position`.
   - Convert R to quaternion — requires special handling because `det(R) = -1` (improper rotation from Y-up world to Y-down OpenCV convention). Fix: if `det(R) < 0`, negate row 2 of R before calling `scipy.Rotation.from_matrix`.
   - Write `cameras.txt`, `images.txt`, `points3D.txt`. Image IDs must match those assigned by the COLMAP database (queried via `sqlite3` after extraction — filesystem order ≠ camera list order).
4. Run `pycolmap.triangulate_points` with locked poses to produce the sparse cloud.

**Dense (Stereo SGBM)**

1. For each camera pair where `dot(forward_i, forward_j) > 0.70` (within ~45°), run stereo rectification.
2. Compute a disparity map with `cv2.StereoSGBM`.
3. Filter:
   - `disparity > 3.0` — rejects near-zero disparity (far background, unreliable)
   - `depth < baseline × 10` — caps maximum depth
   - Sobel gradient magnitude > 1.0 — rejects textureless regions (sand, sky, uniform walls)
   - Intersection ROI of both rectified images — removes border padding artifacts
4. Unproject surviving pixels to world coordinates via `cv2.reprojectImageTo3D`.
5. Aggregate all pairs, IQR outlier filter (scale=3.0), optionally constrain to sparse cloud bounding volume.

### Unknown poses — Statue, Fountain

1. `pycolmap.incremental_mapping` (full SfM) to estimate all camera poses and a sparse cloud.
2. `pycolmap.triangulate_points` to densify the sparse cloud.
3. SGBM dense stereo using COLMAP-estimated poses — each image paired with its nearest angular neighbors.

---

## Key Parameters (final values)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `SIFT_NFEATURES` | 10,000 | Features per image |
| `DOT_THRESH` | 0.70 | ~45° max angle between stereo pair cameras |
| `NUM_DISP` | 256 | SGBM disparity search range (multiple of 16) |
| `BLOCK_SIZE` | 11 | SGBM matching block size |
| `TEXTURE_THRESH` | 15.0 | Lower tends to more surface coverage |
| `uniquenessRatio` | 20 | SGBM confidence gate — rejects ambiguous matches |
| `speckleWindowSize` | 200 | Speckle filter — removes isolated wrong-match blobs |
| `max_depth` | baseline × 10 | Per-pair depth cap |

---

## How We Got Here 

### Step 1 — Understanding the problem and first prototype (`box_sparse_visualize.py`, `box_sgbm_visualize.py`)

We started by reading the competition PDF carefully. The key insight was that for Box and Entrance we have exact camera positions and orientations, which means we can cast rays from each camera through each pixel and triangulate where two rays from different cameras intersect — this gives us a 3D point for every matched feature.

The competition PDF provided a C++ function `GetLineEquation` for computing the ray direction from a pixel. We ported this to Python exactly:

```python
def get_line_equation(pixel_row, pixel_col, res_x, res_y,
                      cam_forward, cam_right, cam_up, cam_position):
    coeff_right = 2.0 * (pixel_col - res_x / 2.0 + 0.5) / res_x
    coeff_up    = -2.0 * (pixel_row - res_y / 2.0 + 0.5) / res_y
    coeff_up   *= res_y / res_x
    direction = cam_forward + coeff_right * cam_right + coeff_up * cam_up
    return cam_position, direction / norm(direction)
```

For triangulation we used the skew-line midpoint method (`calculate_3d_point`): find the two points on two rays that are closest to each other, take the midpoint. This is more robust than `cv2.triangulatePoints` when the camera model is custom.

**First bug found:** `build_extrinsics` had wrong signs — `R = [right, up, -forward]` instead of `[right, -up, forward]`. In OpenCV convention camera Y points down (so `-CamUp`) and Z is depth (so `+CamForward`). With the wrong signs, every triangulated point was reflected through the camera — the whole scene appeared behind it.

**First result:** 3,249 sparse 3D points from 66 camera pairs. Structurally correct but very sparse — SIFT only finds points at strong edges and corners.

---

### Step 2 — Dense stereo with SGBM (`dense_box.py`)

Sparse ray-casting gives a few thousand points. For a full point cloud we need every pixel reconstructed. Without CUDA (no GPU on this machine), COLMAP's `patch_match_stereo` is unavailable. We implemented dense stereo with OpenCV's `StereoSGBM` — a CPU-based Semi-Global Block Matching algorithm.

**How SGBM works:** Given two rectified images of the same scene from slightly different positions, SGBM scans each row and finds the best horizontal shift (disparity) that matches a pixel in the left image to the right image. Disparity is inversely proportional to depth: `depth = fx × baseline / disparity`.

**COLMAP for Statue/Fountain:** For the datasets without known poses, we called COLMAP's Structure-from-Motion pipeline via subprocess to estimate camera positions. This ran into several environment issues because COLMAP was installed via snap (broken libpthread under snap) — fixed by passing a clean minimal environment dict to subprocess. Also needed `QT_QPA_PLATFORM=offscreen` for headless rendering and `--SiftExtraction.use_gpu 0` for CPU-only mode.

**First dense result:** ~3 million points for Box. Very noisy — depth cap of `baseline × 200` was far too large (produced depths at ±50,000 units when the scene is only ~400 units away from cameras). The visualization showed massive outliers at ±4000–6000 that made the real scene invisible.

---

### Step 3 — Diagnosing and fixing the noise (`pipeline_main.py`)

We saved visualizations after every change. Looking at `box_dense_vis.png` and `box_sparse_vis.png` revealed what was actually happening:

- **Box sparse:** clearly shows a rectangle in the top view — the four corners of the box are visible. Correct geometry, just not dense.
- **Box dense:** the box is there in the side view, but surrounded by a huge noise cloud. The scale was wrong because outliers dominated the IQR statistics.

**Fixes that worked:**

1. **Tighter depth cap:** `baseline × 200 → baseline × 15-20`. The box scene is about 350 units from the cameras, baseline between adjacent cameras is ~200, so max useful depth is ~1000. This cut the ±50,000 outliers immediately.

2. **Reprojection error gate for sparse:** After triangulating a 3D point from two rays, reproject it back into both cameras and check the pixel error. If the point reprojects more than 2px from the original keypoint, reject it. This cut sparse noise by ~40%.

3. **ROI mask for dense:** `cv2.stereoRectify` with `alpha=1` gives valid image regions for both cameras after rectification. Restricting to their intersection removes border padding where SGBM produces garbage matches.

4. **Texture mask:** Compute Sobel gradient magnitude on the rectified left image. Only keep pixels above a threshold. This eliminated hallucinated points on the sandy floor (Box) and sky (Entrance) — uniform regions where SGBM matches randomly.

5. **`filter_dense_by_sparse`:** Compute the bounding box of the sparse COLMAP cloud, expand it by a factor, and keep only dense points within that volume. The sparse cloud correctly outlines the real scene; this rejected points in regions where COLMAP found nothing.

**What didn't work:**

- **Unlimiting SIFT features (`nfeatures=0`) and lowering `contrastThreshold`:** Tried to get more keypoints on the plain wooden box faces. Made things significantly worse — too many weak features produced bad matches that scattered the sparse cloud. Reverted immediately.

- **All-pairs dense stereo (66 pairs for 12 cameras):** Tried using every possible camera pair. Cameras more than ~60° apart have a baseline so large that the scene disparity exceeds SGBM's search range, producing completely wrong depth estimates placed far outside the real scene. Diagonal noise streaks appeared everywhere.

- **Sequential COLMAP matching → reverted to exhaustive:** Sequential matching registered more Statue images initially but exhaustive produced better geometry in the final pipeline.

---

### Step 4 — Multi-view geometric consistency experiment (`pipeline_colmap_consistent.py`)

Inspired by what makes COLMAP's dense reconstruction good: it doesn't just compute one depth map per image pair, it checks that each depth map is geometrically consistent with several others. The algorithm:

1. For each reference camera, compute SGBM depth maps using its N nearest neighbors.
2. Unproject each depth map to world coordinates.
3. Reproject each world point back into the reference camera's depth map.
4. Keep only pixels where at least `MIN_CONSISTENT` reprojected depths agree within a tolerance.

This produced 10,867 clean, geometrically verified points for Box — much less noise than raw SGBM. However, it was also much slower and missed large parts of the scene because the consistency requirement was strict. We kept the code in `pipeline_colmap_consistent.py` as a reference but did not use it in the final solution.

---

### Step 5 — Switching to pycolmap Python API (`pipeline_dario.py`)

Up to this point, COLMAP was called via `subprocess` with shell commands. This was fragile: snap environment issues, path problems, needing to parse text files after each step. We switched to the `pycolmap` Python API which runs COLMAP directly in-process.

**Migration summary:**
- `subprocess` calls → `pycolmap.extract_features`, `pycolmap.match_exhaustive`, `pycolmap.incremental_mapping`, `pycolmap.triangulate_points`
- Parsing `points3D.txt` → reading directly from `Reconstruction.points3D`  
- `model_converter` step → `recon.write_text()`

**Bug encountered: `ValueError: Non-positive determinant`**

`scipy.spatial.transform.Rotation.from_matrix` requires a proper rotation matrix (determinant = +1). Our `build_extrinsics` produces `R = [right, -up, forward]` which has `det(R) = -1` — it's an improper rotation because negating the "up" row flips the handedness of the coordinate system.

Fix: before converting to quaternion, check if `det(R) < 0` and if so negate row 2:

```python
def rot_to_quat(R):
    R = np.array(R, dtype=float)
    if np.linalg.det(R) < 0:
        R[2] = -R[2]
    q = st.Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return q[3], q[0], q[1], q[2]             # → qw, qx, qy, qz
```

**Bug encountered: `ValueError: Check failed: existing_frame.DataIds() == frame.DataIds()`**

COLMAP's feature extraction assigns image IDs based on filesystem sort order. On this machine, `box12.png` came first alphabetically and got ID 1. Our code was writing `images.txt` with IDs 1–12 based on the camera list index (box1.png → ID 1), causing a mismatch that crashed triangulation.

Fix: after calling `extract_features`, query the database directly to get the actual IDs:

```python
import sqlite3
conn = sqlite3.connect(str(db))
name_to_db_id = {name: iid for iid, name in conn.execute("SELECT image_id, name FROM images")}
conn.close()
```

Note: `pycolmap.Database()` was tried first but is abstract and cannot be instantiated — the `sqlite3` approach was the working solution.

**Bug encountered: COLMAP refining focal length to wrong value**

For Statue, COLMAP's bundle adjustment was changing the focal length from the known `fy=960` (from `K.txt`) to `fy=3089`. This completely broke the `Q` matrix in `cv2.stereoRectify`, causing all dense pairs to return 0 points.

Fix: `map_opts.ba_refine_focal_length = (K is None)` — only let COLMAP refine the focal length when we don't already have a calibrated K.

---

### Step 6 — The fog ring problem and baseline-disparity geometry

After the pycolmap migration, running Box dense showed a characteristic "fog ring" pattern: the actual box was visible at the center, but surrounded by a diffuse ring of wrong points extending ±400 units in all directions.

**Root cause:** cameras orbit the box at radius ~350 units. For cameras 35° apart, the baseline is roughly `2 × 350 × sin(17.5°) ≈ 210` units. The scene depth to the box is ~350 units. Expected disparity:

```
expected_disparity = fx × baseline / depth = 960 × 210 / 350 ≈ 577 pixels
```

Our `NUM_DISP` was set to 512. When the expected disparity exceeds `NUM_DISP`, SGBM cannot find the correct match and returns a spurious disparity — placing the 3D point at a completely wrong position. These wrong positions formed the fog ring.

**Fix:** three-pronged approach:
1. `DOT_THRESH = 0.82` — only pairs within ~35° (baseline stays ≤ ~200 units where disparity fits in range)
2. `NUM_DISP = 576` — slightly raised to cover the expected ~550px disparity
3. `disp > 3.0` and `max_depth = baseline × 5` — gate out near-zero disparities (fog at infinity) and implausible depths

This eliminated the fog ring. The trade-off: fewer pairs → less scene coverage.

---

### Step 7 — Balancing coverage vs. noise (`submission.py`)

With the fog fix, the scene was geometrically correct but only showed the part of the scene visible in close-angle pairs. The remaining tuning was about widening coverage without reintroducing noise:

**Widening coverage (DOT_THRESH 0.82 → 0.70):**
Allows pairs within ~45° instead of 35°. More pairs = more scene faces covered. To handle the larger baselines, `NUM_DISP` was raised from 576 to 768.

**Deeper geometry (max_depth baseline × 5 → × 10):**
The box scene has geometry at various depths. The tight × 5 cap was cutting legitimate far-wall points.

**Less aggressive outlier filtering (iqr_scale 1.5 → 3.0):**
IQR at scale 1.5 cuts points more than 1.5 interquartile ranges from the median — aggressive enough to prune peripheral real geometry. Scale 3.0 keeps more of the scene edges.

**More surface coverage (TEXTURE_THRESH 2.0 → 1.0):**
Lowered gradient threshold captures lower-texture surfaces that still have enough signal for SGBM.

---

## Running the Final Solution

```bash
# Install dependencies
uv sync

# Run a single dataset
uv run submission.py Box
uv run submission.py Statue

# Sparse or dense only
uv run submission.py Box --sparse
uv run submission.py Box --dense

Running all scenes takes about 25 - 30 minutes on 18 core CPU (unfortunately we didn't have CUDA)
```

Outputs written to `output/`:

| File | Content |
|------|---------|
| `<prefix>_dense_points.txt` | **Submission file** — one `x y z` per line |
| `<prefix>_sparse_points.txt` | Sparse triangulated cloud |
| `<prefix>_dense_points.ply` | Colored cloud for MeshLab |
| `<prefix>_dense_points.xyz` | Colored cloud for CloudCompare |
| `<prefix>_dense_vis.png` | 3-panel visualization (3D view, top view XY, side view XZ) |

---

## Dependencies

- Python ≥ 3.13
- `opencv-python-headless` — SIFT, SGBM, stereo rectification
- `numpy`, `scipy` — linear algebra, rotation conversions
- `pycolmap` — feature extraction, matching, SfM, triangulation
- `matplotlib` — visualization
- `open3d` - for point cloud rendering

```bash
uv sync
# or
pip install opencv-python-headless numpy scipy matplotlib pycolmap
```

## Room for improvements

- Implementing rougher noise filtering and expanding scene coverage
- Adding more feature numbers (GPU required)

## Running open3d visualizator

```bash
uv run visualizer.py Box (Statue, Entrance, Fountain)

if it fails try
LIBGL_ALWAYS_SOFTWARE=1 uv run visualizer.py Box

or force X11 explicitly for GLFW:

XDG_SESSION_TYPE=x11 DISPLAY=:0 uv run visualizer.py Statue
```

## Running custom visualization server

```bash
uv run python3 server.py
```
