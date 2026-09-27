#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#include <emscripten.h>
#include <emscripten/fetch.h>
#include <emscripten/html5.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cfloat>

static const char* VERT = R"(#version 300 es
precision highp float;
in vec3 a_pos;
in vec3 a_col;
uniform mat4 u_mvp;
uniform float u_size;
out vec3 v_col;
void main() {
  vec4 clip = u_mvp * vec4(a_pos, 1.0);
  gl_Position = clip;
  float dist = max(-clip.z, 0.1);
  gl_PointSize = clamp(u_size * 120.0 / dist, 0.5, u_size * 4.0);
  v_col = a_col;
})";

static const char* FRAG = R"(#version 300 es
precision mediump float;
in vec3 v_col;
out vec4 fragColor;
void main() {
  fragColor = vec4(v_col, 1.0);
})";

static const char* LINE_VERT = R"(#version 300 es
precision highp float;
in vec3 a_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(a_pos, 1.0); })";

static const char* LINE_FRAG = R"(#version 300 es
precision mediump float;
uniform vec3 u_color;
out vec4 fragColor;
void main() { fragColor = vec4(u_color, 1.0); })";

struct Mat4 { float m[16] = {}; };

static Mat4 perspective(float fov, float aspect, float near, float far) {
  float f = 1.0f / tanf(fov * 0.5f), nf = 1.0f / (near - far);
  Mat4 o;
  o.m[0]  = f / aspect;
  o.m[5]  = f;
  o.m[10] = (far + near) * nf;
  o.m[11] = -1.0f;
  o.m[14] = 2.0f * far * near * nf;
  return o;
}

static Mat4 lookAt(float ex, float ey, float ez,
                   float cx, float cy, float cz) {
  float fx = ex-cx, fy = ey-cy, fz = ez-cz;
  float fl = sqrtf(fx*fx+fy*fy+fz*fz); fx/=fl; fy/=fl; fz/=fl;
  float rx = -fz, ry = 0.0f, rz = fx;
  float rl = sqrtf(rx*rx+rz*rz); rx/=rl; rz/=rl;
  float ux = fy*rz-fz*ry, uy = fz*rx-fx*rz, uz = fx*ry-fy*rx;
  Mat4 o;
  o.m[0]=rx; o.m[4]=ry; o.m[8] =rz; o.m[12]=-(rx*ex+ry*ey+rz*ez);
  o.m[1]=ux; o.m[5]=uy; o.m[9] =uz; o.m[13]=-(ux*ex+uy*ey+uz*ez);
  o.m[2]=fx; o.m[6]=fy; o.m[10]=fz; o.m[14]=-(fx*ex+fy*ey+fz*ez);
  o.m[15]=1.0f;
  return o;
}

static Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 c;
  for (int col = 0; col < 4; col++)
    for (int row = 0; row < 4; row++) {
      float s = 0;
      for (int k = 0; k < 4; k++) s += a.m[row+k*4] * b.m[k+col*4];
      c.m[row+col*4] = s;
    }
  return c;
}

static GLuint compileShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[512]; glGetShaderInfoLog(s,512,nullptr,log); printf("Shader: %s\n",log); }
  return s;
}

static GLuint makeProgram(const char* vert, const char* frag) {
  GLuint vs = compileShader(GL_VERTEX_SHADER, vert);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, frag);
  GLuint p  = glCreateProgram();
  glAttachShader(p, vs); glAttachShader(p, fs);
  glLinkProgram(p);
  glDeleteShader(vs); glDeleteShader(fs);
  return p;
}

static SDL_Window*   g_win    = nullptr;
static SDL_GLContext g_ctx    = nullptr;
static GLuint        g_prog   = 0;
static GLuint        g_vao    = 0;
static GLuint        g_posBuf = 0;
static GLuint        g_colBuf = 0;
static int           g_count  = 0;
static int           g_locMvp = -1, g_locSz = -1;

static GLuint g_lineProg  = 0;
static int    g_lineLocMvp = -1, g_lineLocCol = -1;
static GLuint g_gridVao = 0, g_gridBuf = 0;
static GLuint g_axisVao = 0, g_axisBuf = 0;
static int    g_gridCount = 0;
static float  g_cloudMinY = 0.0f;
static float  g_lastHelpR = -1.0f;

static float g_az = 0.4f, g_el = 0.25f, g_r = 10.0f;
static float g_tx = 0.0f, g_ty = 0.0f,  g_tz = 0.0f;
static float g_sz = 1.0f;

static bool     g_ldown = false, g_rdown = false;
static int      g_mx = 0,  g_my = 0;
static int      g_winW = 800, g_winH = 600;
static uint32_t g_lastInteract = 0;
static uint32_t g_lastRender   = 0;

static void uploadCloud(const float* pos, const uint8_t* col, int count) {
  glBindVertexArray(g_vao);

  glBindBuffer(GL_ARRAY_BUFFER, g_posBuf);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(count * 3 * 4), pos, GL_STATIC_DRAW);
  GLint pl = glGetAttribLocation(g_prog, "a_pos");
  glEnableVertexAttribArray(pl);
  glVertexAttribPointer(pl, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, g_colBuf);
  glBufferData(GL_ARRAY_BUFFER, count * 3, col, GL_STATIC_DRAW);
  GLint cl = glGetAttribLocation(g_prog, "a_col");
  glEnableVertexAttribArray(cl);
  glVertexAttribPointer(cl, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, nullptr);

  int step = std::max(1, count / 5000);
  float cx=0,cy=0,cz=0; int n=0;
  for (int i=0; i<count; i+=step) { cx+=pos[i*3]; cy+=pos[i*3+1]; cz+=pos[i*3+2]; n++; }
  g_tx=cx/n; g_ty=cy/n; g_tz=cz/n;
  float maxD=0;
  for (int i=0; i<count; i+=step) {
    float dx=pos[i*3]-g_tx, dy=pos[i*3+1]-g_ty, dz=pos[i*3+2]-g_tz;
    float d=sqrtf(dx*dx+dy*dy+dz*dz); if(d>maxD) maxD=d;
  }
  float minY = FLT_MAX;
  for (int i = 0; i < count; i += step)
    if (pos[i*3+1] < minY) minY = pos[i*3+1];
  g_cloudMinY = minY;

  g_r=maxD*2.2f; g_az=0.4f; g_el=0.25f;
  g_count=count;
  g_lastHelpR = -1.0f;
  g_lastInteract = SDL_GetTicks();
  printf("[viewer] uploaded %d points\n", count);
}

static void onFetchOK(emscripten_fetch_t* fetch) {
  const uint8_t* data = (const uint8_t*)fetch->data;
  uint32_t count;
  memcpy(&count, data, 4);
  const float*   pos = (const float*)(data + 4);
  const uint8_t* col = data + 4 + count * 12;
  uploadCloud(pos, col, (int)count);
  emscripten_fetch_close(fetch);
  EM_ASM({ if (window._onCloudLoaded) window._onCloudLoaded($0); }, count);
}

static void onFetchErr(emscripten_fetch_t* fetch) {
  printf("[viewer] fetch error %d\n", fetch->status);
  emscripten_fetch_close(fetch);
  EM_ASM({ if (window._onCloudLoaded) window._onCloudLoaded(0); });
}

static float niceStep(float v) {
  if (v <= 0.0f) return 1.0f;
  float p = powf(10.0f, floorf(log10f(v)));
  float r = v / p;
  if (r < 2.0f) return p;
  if (r < 5.0f) return p * 2.0f;
  return p * 5.0f;
}

static void buildHelpers() {
  float spacing = niceStep(g_r * 0.3f);
  int   n       = std::min((int)(g_r * 1.5f / spacing) + 1, 25);
  float ext     = spacing * n;
  float cx      = g_tx, cy = g_cloudMinY, cz = g_tz;

  std::vector<float> gridV;
  gridV.reserve((n*2+1) * 4 * 3);
  for (int i = -n; i <= n; i++) {
    float x = cx + i * spacing;
    gridV.push_back(x); gridV.push_back(cy); gridV.push_back(cz - ext);
    gridV.push_back(x); gridV.push_back(cy); gridV.push_back(cz + ext);
  }
  for (int i = -n; i <= n; i++) {
    float z = cz + i * spacing;
    gridV.push_back(cx - ext); gridV.push_back(cy); gridV.push_back(z);
    gridV.push_back(cx + ext); gridV.push_back(cy); gridV.push_back(z);
  }
  g_gridCount = (int)(gridV.size() / 3);

  GLint pl = glGetAttribLocation(g_lineProg, "a_pos");
  glBindVertexArray(g_gridVao);
  glBindBuffer(GL_ARRAY_BUFFER, g_gridBuf);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(gridV.size() * 4), gridV.data(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(pl);
  glVertexAttribPointer(pl, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  float al = spacing * 3.0f;
  float ax = cx, ay = cy, az = cz;
  float axisV[] = {
    ax,    ay, az,    ax+al, ay,    az,
    ax,    ay, az,    ax,    ay+al, az,
    ax,    ay, az,    ax,    ay,    az+al,
  };
  glBindVertexArray(g_axisVao);
  glBindBuffer(GL_ARRAY_BUFFER, g_axisBuf);
  glBufferData(GL_ARRAY_BUFFER, sizeof(axisV), axisV, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(pl);
  glVertexAttribPointer(pl, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  g_lastHelpR = g_r;
}

static void frame() {
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    switch (e.type) {
      case SDL_MOUSEBUTTONDOWN:
        g_ldown |= e.button.button == SDL_BUTTON_LEFT;
        g_rdown |= e.button.button == SDL_BUTTON_RIGHT;
        g_mx = e.button.x; g_my = e.button.y;
        g_lastInteract = SDL_GetTicks();
        break;
      case SDL_MOUSEBUTTONUP:
        if (e.button.button == SDL_BUTTON_LEFT)  g_ldown = false;
        if (e.button.button == SDL_BUTTON_RIGHT) g_rdown = false;
        break;
      case SDL_MOUSEMOTION:
        if (g_ldown || g_rdown) {
          float ddx = (float)(e.motion.x - g_mx);
          float ddy = (float)(e.motion.y - g_my);
          if (g_ldown) {
            g_az -= ddx * 0.005f;
            g_el  = fmaxf(-1.55f, fminf(1.55f, g_el + ddy * 0.005f));
          }
          if (g_rdown) {
            float rx  =  cosf(g_az);
            float rz  =  sinf(g_az);
            float upx = -sinf(g_el)*sinf(g_az);
            float upy =  cosf(g_el);
            float upz = -sinf(g_el)*cosf(g_az);
            float s   =  g_r * 0.002f;
            g_tx -= (ddx * rx - ddy * upx) * s;
            g_ty -= (ddy * upy) * s;
            g_tz -= (ddx * rz - ddy * upz) * s;
          }
          g_lastInteract = SDL_GetTicks();
        }
        g_mx = e.motion.x; g_my = e.motion.y;
        break;
      case SDL_MOUSEWHEEL:
        g_r *= 1.0f - e.wheel.y * 0.1f;
        g_r  = fmaxf(0.0001f, g_r);
        g_lastInteract = SDL_GetTicks();
        break;
      case SDL_WINDOWEVENT:
        if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
          g_winW = e.window.data1; g_winH = e.window.data2;
          glViewport(0, 0, g_winW, g_winH);
          g_lastInteract = SDL_GetTicks();
        }
        break;
    }
  }

  uint32_t now = SDL_GetTicks();
  bool active = (now - g_lastInteract) < 400;
  if (!active) {
    if (now - g_lastRender < 160) return;
  }
  g_lastRender = now;

  if (g_lastHelpR < 0.0f || fabsf(g_r - g_lastHelpR) / fmaxf(g_r, 1e-6f) > 0.05f)
    buildHelpers();

  glClearColor(0.059f, 0.067f, 0.090f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  float aspect = (g_winH > 0) ? (float)g_winW / g_winH : 1.0f;
  float ex = g_tx + g_r*cosf(g_el)*sinf(g_az);
  float ey = g_ty + g_r*sinf(g_el);
  float ez = g_tz + g_r*cosf(g_el)*cosf(g_az);
  Mat4 proj = perspective(M_PI/3.0f, aspect, g_r*0.001f, g_r*5000.0f);
  Mat4 view = lookAt(ex, ey, ez, g_tx, g_ty, g_tz);
  Mat4 mvp  = mul(proj, view);

  glUseProgram(g_lineProg);
  glUniformMatrix4fv(g_lineLocMvp, 1, GL_FALSE, mvp.m);
  glUniform3f(g_lineLocCol, 0.20f, 0.26f, 0.38f);
  glBindVertexArray(g_gridVao);
  glDrawArrays(GL_LINES, 0, g_gridCount);

  glBindVertexArray(g_axisVao);
  glUniform3f(g_lineLocCol, 0.90f, 0.18f, 0.18f); glDrawArrays(GL_LINES, 0, 2);
  glUniform3f(g_lineLocCol, 0.18f, 0.85f, 0.28f); glDrawArrays(GL_LINES, 2, 2);
  glUniform3f(g_lineLocCol, 0.18f, 0.48f, 0.92f); glDrawArrays(GL_LINES, 4, 2);

  if (g_count > 0) {
    glUseProgram(g_prog);
    glUniformMatrix4fv(g_locMvp, 1, GL_FALSE, mvp.m);
    glUniform1f(g_locSz, g_sz);
    glBindVertexArray(g_vao);
    glDrawArrays(GL_POINTS, 0, g_count);
  }

  SDL_GL_SwapWindow(g_win);
}

extern "C" {

EMSCRIPTEN_KEEPALIVE void load_cloud(float* pos, uint8_t* col, int count) {
  uploadCloud(pos, col, count);
}

EMSCRIPTEN_KEEPALIVE void fetch_cloud(const char* url) {
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
  attr.onsuccess  = onFetchOK;
  attr.onerror    = onFetchErr;
  emscripten_fetch(&attr, url);
}

EMSCRIPTEN_KEEPALIVE void set_point_size(float s) { g_sz = s; g_lastInteract = SDL_GetTicks(); }
EMSCRIPTEN_KEEPALIVE int  get_point_count()        { return g_count; }
EMSCRIPTEN_KEEPALIVE void clear_cloud()             { g_count = 0; }
EMSCRIPTEN_KEEPALIVE void set_canvas_size(int w, int h) {
  SDL_SetWindowSize(g_win, w, h);
  glViewport(0, 0, w, h);
  g_winW = w; g_winH = h;
  g_lastInteract = SDL_GetTicks();
}

} // extern "C"

int main() {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

  g_win = SDL_CreateWindow("PointCloud", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                            800, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
  g_ctx = SDL_GL_CreateContext(g_win);

  g_prog   = makeProgram(VERT, FRAG);
  g_locMvp = glGetUniformLocation(g_prog, "u_mvp");
  g_locSz  = glGetUniformLocation(g_prog, "u_size");

  g_lineProg   = makeProgram(LINE_VERT, LINE_FRAG);
  g_lineLocMvp = glGetUniformLocation(g_lineProg, "u_mvp");
  g_lineLocCol = glGetUniformLocation(g_lineProg, "u_color");

  glGenVertexArrays(1, &g_vao);
  glGenBuffers(1, &g_posBuf);
  glGenBuffers(1, &g_colBuf);
  glGenVertexArrays(1, &g_gridVao);
  glGenBuffers(1, &g_gridBuf);
  glGenVertexArrays(1, &g_axisVao);
  glGenBuffers(1, &g_axisBuf);
  glEnable(GL_DEPTH_TEST);

  emscripten_set_main_loop(frame, 0, 0);
  return 0;
}
