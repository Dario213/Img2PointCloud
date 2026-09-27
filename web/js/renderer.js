/* globals createREGL */

const VERT = `
  precision highp float;
  attribute vec3 position;
  attribute vec3 color;
  uniform mat4 mvp;
  uniform float pointSize;
  varying vec3 vColor;
  void main() {
    vec4 clip = mvp * vec4(position, 1.0);
    gl_Position = clip;
    float dist = clamp(-clip.z, 0.1, 1e6);
    gl_PointSize = clamp(pointSize * 120.0 / dist, 1.0, 8.0);
    vColor = color / 255.0;
  }
`;

const FRAG = `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }
`;

class PointCloudRenderer {
  constructor(canvas) {
    this.regl = createREGL({
      canvas,
      attributes: { antialias: false, alpha: false, depth: true },
      extensions: [],
    });

    this._posBuf  = null;
    this._colBuf  = null;
    this._count   = 0;
    this._pointSize = 2.0;

    this._draw = this.regl({
      vert: VERT,
      frag: FRAG,
      attributes: {
        position: () => this._posBuf,
        color:    () => this._colBuf,
      },
      uniforms: {
        mvp:       this.regl.prop('mvp'),
        pointSize: () => this._pointSize,
      },
      count:     () => this._count,
      primitive: 'points',
      depth: { enable: true, mask: true },
    });
  }

  load(positions, colors, count) {
    if (this._posBuf) this._posBuf.destroy();
    if (this._colBuf) this._colBuf.destroy();

    this._posBuf = this.regl.buffer({ data: positions, type: 'float', usage: 'static' });
    this._colBuf = this.regl.buffer({ data: colors,    type: 'uint8', usage: 'static' });
    this._count  = count;
  }

  clear() {
    this._count = 0;
  }

  setPointSize(s) { this._pointSize = s; console.log("set point size to: ", s);}

  render(mvp) {
    const regl = this.regl;
    regl.poll();
    regl.clear({ color: [0.059, 0.067, 0.090, 1], depth: 1 });
    if (this._count > 0) this._draw({ mvp: Array.from(mvp) });
  }

  destroy() { this.regl.destroy(); }
}
