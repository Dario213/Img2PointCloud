import asyncio
import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

OUTPUT_DIR = Path("output")
IMAGES_DIR = Path("StemGames2026_ProjectTask/TestImages")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

DATASETS = ["Box", "Entrance", "Statue", "Fountain"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# ── Point cloud loading & caching ─────────────────────────────────────────────

_cloud_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _parse_ply_ascii(path: Path):
    with open(path) as f:
        n = 0
        for line in f:
            if "element vertex" in line:
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f, max_rows=n, dtype=np.float32)
    return data[:, :3], data[:, 3:6].astype(np.uint8)


def _load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    key = str(path)
    if key in _cloud_cache:
        return _cloud_cache[key]

    cache = path.with_suffix(".npz")
    if cache.exists():
        d = np.load(cache)
        pos, col = d["pos"], d["col"]
    else:
        pos, col = _parse_ply_ascii(path)
        np.savez(cache, pos=pos, col=col)

    _cloud_cache[key] = pos, col
    return pos, col


def _pack_binary(pos: np.ndarray, col: np.ndarray) -> bytes:
    """uint32 count | float32 XYZ ... | uint8 RGB ..."""
    count = np.array([len(pos)], dtype=np.uint32)
    return count.tobytes() + pos.astype(np.float32).tobytes() + col.astype(np.uint8).tobytes()


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    result = []
    for name in DATASETS:
        p = name.lower()
        result.append({
            "name": name,
            "dense": (OUTPUT_DIR / f"{p}_dense_points.ply").exists(),
            "sparse": (OUTPUT_DIR / f"{p}_sparse_points.ply").exists(),
        })
    return result


@app.get("/api/pointcloud/{name}")
def get_pointcloud(name: str, type: str = "dense", max_points: int = 3_000_000):
    path = OUTPUT_DIR / f"{name.lower()}_{type}_points.ply"
    if not path.exists():
        raise HTTPException(404, "not found")

    pos, col = _load_cloud(path)

    if len(pos) > max_points:
        idx = np.random.choice(len(pos), max_points, replace=False)
        pos, col = pos[idx], col[idx]

    return Response(_pack_binary(pos, col), media_type="application/octet-stream")


@app.get("/api/images/{name}")
def list_images(name: str):
    d = IMAGES_DIR / name
    if not d.exists():
        raise HTTPException(404)
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)


@app.get("/api/image/{name}/{filename}")
def get_image(name: str, filename: str):
    path = IMAGES_DIR / name / filename
    if not path.exists():
        raise HTTPException(404)
    mt = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return Response(path.read_bytes(), media_type=mt)


@app.get("/api/sift/{name}/{filename}")
def get_sift(name: str, filename: str):
    path = IMAGES_DIR / name / filename
    if not path.exists():
        raise HTTPException(404)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    kps, _ = cv2.SIFT_create(nfeatures=2000).detectAndCompute(img, None)
    return [
        {"x": kp.pt[0], "y": kp.pt[1], "size": kp.size, "angle": kp.angle}
        for kp in kps
    ]


# ── Upload + process ──────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(
    images: list[UploadFile] = File(...),
    camera_file: UploadFile = File(None),
):
    uid = uuid.uuid4().hex[:8]
    work = UPLOADS_DIR / uid
    work.mkdir()

    for img in images:
        (work / img.filename).write_bytes(await img.read())

    has_cameras = camera_file is not None
    if has_cameras:
        (work / "cameras.txt").write_bytes(await camera_file.read())

    out_prefix = f"upload_{uid}"

    async def stream():
        yield f"data: {json.dumps({'status': 'started', 'id': uid})}\n\n"

        cmd = [
            "uv", "run", "python", "pipeline_custom.py",
            str(work), out_prefix,
            "--cameras" if has_cameras else "--no-cameras",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            msg = line.decode().strip()
            if msg:
                yield f"data: {json.dumps({'status': 'log', 'msg': msg})}\n\n"

        await proc.wait()
        ply = OUTPUT_DIR / f"{out_prefix}_dense_points.ply"
        if proc.returncode == 0 and ply.exists():
            yield f"data: {json.dumps({'status': 'done', 'key': out_prefix})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'error'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/pointcloud_upload/{key}")
def get_upload_cloud(key: str, max_points: int = 10_000_000):
    path = OUTPUT_DIR / f"{key}_dense_points.ply"
    if not path.exists():
        raise HTTPException(404)
    pos, col = _load_cloud(path)
    if len(pos) > max_points:
        idx = np.random.choice(len(pos), max_points, replace=False)
        pos, col = pos[idx], col[idx]
    return Response(_pack_binary(pos, col), media_type="application/octet-stream")


# Serve web UI last so API routes take priority
app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
