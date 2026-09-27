// ─── Wait for WASM ────────────────────────────────────────────────────────────

function onWasmReady(fn) {
  if (window._wasmReady) fn();
  else window.addEventListener('wasmready', fn, { once: true });
}

// ─── Load point cloud — C++ fetches directly, no JS→WASM copy ───────────────

function loadCloud(url, label) {
  showLoading(true);
  // C++ calls window._onCloudLoaded(count) when done
  window._onCloudLoaded = (count) => {
    showLoading(false);
    document.getElementById('stat-pts').textContent   = count > 0 ? count.toLocaleString() : 'error';
    document.getElementById('stat-label').textContent = label;
    applyPointSize(_pointSize);
    window._onCloudLoaded = null;
  };
  Module.ccall('fetch_cloud', null, ['string'], [url]);
}

function showLoading(on) {
  document.getElementById('loading').style.display = on ? 'block' : 'none';
}

// ─── Resize: keep canvas pixel-perfect and notify C++ ────────────────────────

function syncCanvasSize() {
  const canvas = document.getElementById('canvas');
  const wrap   = document.getElementById('canvas-wrap');
  const dpr    = window.devicePixelRatio || 1;
  const w = Math.floor(wrap.clientWidth  * dpr);
  const h = Math.floor(wrap.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width  = w;
    canvas.height = h;
    if (window._wasmReady) {
      Module.ccall('set_canvas_size', null, ['number', 'number'], [w, h]);
    }
  }
}

new ResizeObserver(syncCanvasSize).observe(document.getElementById('canvas-wrap'));

document.getElementById('sidebar-toggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('collapsed');
  // Wait for the CSS transition to finish before syncing the canvas size
  setTimeout(syncCanvasSize, 230);
});
document.getElementById('canvas').addEventListener('contextmenu', e => e.preventDefault());

// ─── FPS counter ──────────────────────────────────────────────────────────────

let _frames = 0, _lastT = performance.now();
function countFPS() {
  _frames++;
  const now = performance.now();
  if (now - _lastT >= 1000) {
    document.getElementById('stat-fps').textContent = _frames;
    _frames = 0; _lastT = now;
  }
  requestAnimationFrame(countFPS);
}
requestAnimationFrame(countFPS);

// ─── Dataset sidebar ──────────────────────────────────────────────────────────

async function initDatasets() {
  const list      = await fetch('/api/datasets').then(r => r.json());
  const container = document.getElementById('dataset-list');
  container.innerHTML = '';

  for (const ds of list) {
    const item = document.createElement('div');
    item.className = 'dataset-item';

    const name = document.createElement('div');
    name.className = 'dataset-name';
    name.textContent = ds.name;

    const btns = document.createElement('div');
    btns.className = 'dataset-btns';

    btns.append(
      makeBtn('Dense',  ds.dense,  btn => doLoad(btn, `/api/pointcloud/${ds.name}?type=dense`,  `${ds.name} · Dense`,  ds.name)),
      makeBtn('Sparse', ds.sparse, btn => doLoad(btn, `/api/pointcloud/${ds.name}?type=sparse`, `${ds.name} · Sparse`, ds.name)),
    );
    item.append(name, btns);
    container.appendChild(item);
  }
}

function doLoad(btn, url, label, datasetName) {
  setActive(btn);
  loadCloud(url, label);
  loadImageList(datasetName);
}

function makeBtn(label, enabled, handler) {
  const b = document.createElement('button');
  b.className = 'btn';
  b.textContent = label;
  b.disabled = !enabled;
  b.addEventListener('click', () => handler(b));
  return b;
}

let _activeBtn = null;
function setActive(btn) {
  if (_activeBtn) _activeBtn.classList.remove('active');
  btn.classList.add('active');
  _activeBtn = btn;
}

// ─── Image list + SIFT ────────────────────────────────────────────────────────

async function loadImageList(name) {
  const section = document.getElementById('images-section');
  const list    = document.getElementById('image-list');
  list.innerHTML = '';
  section.style.display = 'block';
  document.getElementById('images-title').textContent = `${name} — Images (click for SIFT)`;

  const images = await fetch(`/api/images/${name}`).then(r => r.json());
  for (const filename of images) {
    const img = document.createElement('img');
    img.className = 'img-thumb';
    img.src = `/api/image/${name}/${filename}`;
    img.title = filename;
    img.addEventListener('click', () => showSIFT(name, filename));
    list.appendChild(img);
  }
}

// ─── SIFT modal ───────────────────────────────────────────────────────────────

const siftModal  = document.getElementById('sift-modal');
const siftCanvas = document.getElementById('sift-canvas');
const siftCtx    = siftCanvas.getContext('2d');
const siftInfo   = document.getElementById('sift-info');

document.getElementById('sift-close').addEventListener('click', () => siftModal.classList.remove('open'));
siftModal.addEventListener('click', e => { if (e.target === siftModal) siftModal.classList.remove('open'); });

async function showSIFT(dataset, filename) {
  siftInfo.textContent = 'Loading…';
  siftModal.classList.add('open');
  document.getElementById('sift-title').textContent = filename;

  const [kps, img] = await Promise.all([
    fetch(`/api/sift/${dataset}/${filename}`).then(r => r.json()),
    loadImg(`/api/image/${dataset}/${filename}`),
  ]);

  const scale = Math.min(window.innerWidth * 0.7 / img.width, window.innerHeight * 0.7 / img.height, 1);
  siftCanvas.width  = img.width  * scale;
  siftCanvas.height = img.height * scale;
  siftCtx.drawImage(img, 0, 0, siftCanvas.width, siftCanvas.height);

  for (const kp of kps) {
    const x = kp.x * scale, y = kp.y * scale, r = Math.max(kp.size / 2 * scale, 2);
    const a = kp.angle * Math.PI / 180;
    siftCtx.beginPath();
    siftCtx.arc(x, y, r, 0, Math.PI * 2);
    siftCtx.strokeStyle = 'rgba(79,142,247,0.9)';
    siftCtx.lineWidth = 1.5;
    siftCtx.stroke();
    siftCtx.beginPath();
    siftCtx.moveTo(x, y);
    siftCtx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
    siftCtx.strokeStyle = 'rgba(110,231,183,0.9)';
    siftCtx.stroke();
  }
  siftInfo.textContent = `${kps.length} SIFT keypoints  ·  ${filename}`;
}

function loadImg(src) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = src;
  });
}

// ─── Upload modal ─────────────────────────────────────────────────────────────

const uploadModal = document.getElementById('upload-modal');
const progressLog = document.getElementById('progress-log');
let uploadedFiles = [];

document.getElementById('upload-btn').addEventListener('click', () => uploadModal.classList.add('open'));
document.getElementById('upload-close').addEventListener('click', closeUpload);
uploadModal.addEventListener('click', e => { if (e.target === uploadModal) closeUpload(); });

function closeUpload() {
  uploadModal.classList.remove('open');
  uploadedFiles = [];
  document.getElementById('file-list').textContent = '';
  progressLog.className = '';
  progressLog.textContent = '';
  document.getElementById('submit-btn').disabled = false;
}

const dropzone  = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag-over');
  handleFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => handleFiles([...fileInput.files]));

function handleFiles(files) {
  uploadedFiles = files.filter(f => /\.(png|jpg|jpeg)$/i.test(f.name));
  document.getElementById('file-list').textContent =
    uploadedFiles.length ? uploadedFiles.map(f => f.name).join(', ') : 'No images selected';
}

document.getElementById('submit-btn').addEventListener('click', async () => {
  if (!uploadedFiles.length) return alert('Select images first.');
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  progressLog.className = 'visible';
  progressLog.textContent = '';

  const log = msg => { progressLog.textContent += msg + '\n'; progressLog.scrollTop = 9e9; };

  const form = new FormData();
  for (const f of uploadedFiles) form.append('images', f);
  const camFile = document.getElementById('camera-input').files[0];
  if (camFile) form.append('camera_file', camFile);

  log('Uploading…');
  const res    = await fetch('/api/upload', { method: 'POST', body: form });
  const reader = res.body.getReader();
  const dec    = new TextDecoder();
  let resultKey = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of dec.decode(value).split('\n')) {
      if (!line.startsWith('data:')) continue;
      const evt = JSON.parse(line.slice(5));
      if (evt.status === 'log')   log(evt.msg);
      if (evt.status === 'done')  { resultKey = evt.key; log('✓ Done!'); }
      if (evt.status === 'error') log('✗ Pipeline error.');
    }
  }

  if (resultKey) {
    closeUpload();
    await loadCloud(`/api/pointcloud_upload/${resultKey}`, `Upload · ${resultKey}`);
  }
});

// ─── Point size control ───────────────────────────────────────────────────────

let _pointSize = parseFloat(document.getElementById('point-size').value);

function applyPointSize(v) {
  _pointSize = v;
  document.getElementById('point-size-val').textContent = v;
  document.getElementById('point-size').value = v;
  if (window._wasmReady)
    Module.ccall('set_point_size', null, ['number'], [v]);
}

document.getElementById('point-size').addEventListener('input', e => {
  applyPointSize(parseFloat(e.target.value));
});

// ─── Boot ─────────────────────────────────────────────────────────────────────

onWasmReady(async () => {
  syncCanvasSize();
  applyPointSize(_pointSize);
  await initDatasets();
});
