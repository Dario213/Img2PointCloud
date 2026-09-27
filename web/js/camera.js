// Minimal vec3 / mat4 math — no external dependencies
const v3 = {
  sub:  (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
  add:  (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
  scale: (a, s) => [a[0]*s, a[1]*s, a[2]*s],
  dot:  (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
  cross: (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
  norm: (a) => { const l = Math.sqrt(v3.dot(a,a)); return l > 0 ? [a[0]/l,a[1]/l,a[2]/l] : [0,0,0]; },
  len:  (a) => Math.sqrt(v3.dot(a, a)),
};

function lookAt(eye, ctr, up) {
  const z = v3.norm(v3.sub(eye, ctr));
  const x = v3.norm(v3.cross(up, z));
  const y = v3.cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -v3.dot(x,eye), -v3.dot(y,eye), -v3.dot(z,eye), 1,
  ]);
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([
    f/aspect, 0, 0,  0,
    0,        f, 0,  0,
    0, 0, (far+near)*nf, -1,
    0, 0, 2*far*near*nf, 0,
  ]);
}

function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r + k*4] * b[k + c*4];
      o[r + c*4] = s;
    }
  return o;
}

class OrbitCamera {
  constructor(canvas) {
    this.azimuth   = 0.4;
    this.elevation = 0.25;
    this.radius    = 10;
    this.target    = [0, 0, 0];
    this._canvas   = canvas;
    this._bindEvents();
  }

  getEye() {
    const { azimuth: az, elevation: el, radius: r, target: t } = this;
    return [
      t[0] + r * Math.cos(el) * Math.sin(az),
      t[1] + r * Math.sin(el),
      t[2] + r * Math.cos(el) * Math.cos(az),
    ];
  }

  getMVP(w, h) {
    const proj = perspective(Math.PI / 3, w / h, 0.001 * this.radius, 1000 * this.radius);
    const view = lookAt(this.getEye(), this.target, [0, 1, 0]);
    return mul4(proj, view);
  }

  fitToCloud(positions, count) {
    let cx = 0, cy = 0, cz = 0;
    const step = Math.max(1, Math.floor(count / 5000));
    let samples = 0;
    for (let i = 0; i < count; i += step) {
      cx += positions[i*3]; cy += positions[i*3+1]; cz += positions[i*3+2];
      samples++;
    }
    this.target = [cx/samples, cy/samples, cz/samples];

    let maxD = 0;
    for (let i = 0; i < count; i += step) {
      const dx = positions[i*3]   - this.target[0];
      const dy = positions[i*3+1] - this.target[1];
      const dz = positions[i*3+2] - this.target[2];
      maxD = Math.max(maxD, Math.sqrt(dx*dx + dy*dy + dz*dz));
    }
    this.radius = maxD * 2.2;
  }

  _bindEvents() {
    const c = this._canvas;
    let dragging = false, panning = false;
    let lastX = 0, lastY = 0;

    c.addEventListener('mousedown', e => {
      if (e.button === 2) { panning = true; e.preventDefault(); }
      else { dragging = true; }
      lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = false; panning = false; });
    window.addEventListener('mousemove', e => {
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragging) {
        this.azimuth   -= dx * 0.005;
        this.elevation  = Math.max(-1.55, Math.min(1.55, this.elevation + dy * 0.005));
      }
      if (panning) {
        // pan in camera-right and camera-up directions
        const eye = this.getEye();
        const forward = v3.norm(v3.sub(this.target, eye));
        const right   = v3.norm(v3.cross(forward, [0,1,0]));
        const up      = v3.cross(right, forward);
        const scale   = this.radius * 0.001;
        const shift   = v3.add(v3.scale(right, -dx*scale), v3.scale(up, dy*scale));
        this.target   = v3.add(this.target, shift);
      }
    });
    c.addEventListener('wheel', e => {
      this.radius *= 1 + e.deltaY * 0.001;
      this.radius  = Math.max(0.001, this.radius);
      e.preventDefault();
    }, { passive: false });
    c.addEventListener('contextmenu', e => e.preventDefault());
  }
}
