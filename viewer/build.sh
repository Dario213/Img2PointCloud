#!/bin/bash
set -e

source ~/emsdk/emsdk_env.sh

em++ viewer/main.cpp \
  -O3 \
  -s USE_SDL=2 \
  -s USE_WEBGL2=1 \
  -s FULL_ES3=1 \
  -s FETCH=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_FUNCTIONS='["_main","_malloc","_free","_load_cloud","_fetch_cloud","_set_point_size","_get_point_count","_clear_cloud","_set_canvas_size"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8"]' \
  -s NO_EXIT_RUNTIME=1 \
  -s ENVIRONMENT=web \
  -o web/viewer.js

echo "Built: web/viewer.js + web/viewer.wasm"
