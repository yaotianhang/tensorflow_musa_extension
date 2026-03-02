#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${SCRIPT_DIR}"

# -------- defaults --------
BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="Release"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GENERATOR="${GENERATOR:-}"
JOBS="${JOBS:-}"
CMAKE_EXTRA_ARGS=()
CLEAN=0
VERBOSE=0
CONFIGURE_ONLY=0

# -------- helpers --------
log()  { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
err()  { printf '[ERROR] %s\n' "$*" >&2; }

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] [-- <extra cmake args>]

Options:
  -b, --build-dir DIR       Build directory (default: ./build)
  -t, --build-type TYPE     CMake build type (default: Release)
      --python PATH         Python executable (default: python3 or \$PYTHON_BIN)
  -j, --jobs N              Parallel jobs (default: auto-detect)
  -G, --generator NAME      CMake generator (e.g. Ninja, Unix Makefiles)
      --clean               Remove build directory before configure
      --verbose             Verbose build output
      --configure-only      Run CMake configure only (do not build)
  -h, --help                Show this help

Examples:
  ./build.sh --clean -t Debug
  ./build.sh -G Ninja -j 16 -- -DMUSA_PATH=/usr/local/musa
USAGE
}

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  err "Build failed (exit=${exit_code}) at line ${line_no}."
  err "Project root: ${PROJECT_ROOT}"
  err "Build dir: ${BUILD_DIR}"
  exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

command_exists() { command -v "$1" >/dev/null 2>&1; }

resolve_jobs() {
  if [[ -n "${JOBS}" ]]; then
    printf '%s' "${JOBS}"
    return
  fi

  if command_exists nproc; then
    nproc
  elif command_exists getconf; then
    getconf _NPROCESSORS_ONLN
  elif command_exists sysctl; then
    sysctl -n hw.ncpu
  else
    printf '4'
  fi
}

# -------- argument parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    -t|--build-type)
      BUILD_TYPE="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    -j|--jobs)
      JOBS="$2"; shift 2 ;;
    -G|--generator)
      GENERATOR="$2"; shift 2 ;;
    --clean)
      CLEAN=1; shift ;;
    --verbose)
      VERBOSE=1; shift ;;
    --configure-only)
      CONFIGURE_ONLY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      CMAKE_EXTRA_ARGS+=("$@")
      break ;;
    *)
      err "Unknown argument: $1"
      usage
      exit 2 ;;
  esac
done

# -------- preflight checks --------
for tool in cmake "${PYTHON_BIN}"; do
  if ! command_exists "$tool"; then
    err "Required tool not found: $tool"
    exit 127
  fi
done

# Validate TensorFlow import early because CMakeLists.txt depends on tf.sysconfig.
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import tensorflow as tf
print(tf.__version__)
PY
then
  err "TensorFlow is not importable from Python executable: ${PYTHON_BIN}"
  err "Install TensorFlow in that environment or pass --python /path/to/python"
  exit 1
fi

# Normalize build dir to absolute path for clearer logs/debugging.
mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd -- "${BUILD_DIR}" && pwd -P)"

if [[ ${CLEAN} -eq 1 ]]; then
  log "Cleaning build directory: ${BUILD_DIR}"
  rm -rf -- "${BUILD_DIR}"
  mkdir -p -- "${BUILD_DIR}"
fi

PARALLEL_JOBS="$(resolve_jobs)"

# -------- configure --------
log "Project root : ${PROJECT_ROOT}"
log "Build dir    : ${BUILD_DIR}"
log "Build type   : ${BUILD_TYPE}"
log "Python       : $(command -v "${PYTHON_BIN}")"
log "Jobs         : ${PARALLEL_JOBS}"
[[ -n "${GENERATOR}" ]] && log "Generator    : ${GENERATOR}"

CMAKE_CMD=(cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DPYTHON_EXECUTABLE="$(command -v "${PYTHON_BIN}")"
)

if [[ -n "${GENERATOR}" ]]; then
  CMAKE_CMD+=( -G "${GENERATOR}" )
fi

if [[ ${#CMAKE_EXTRA_ARGS[@]} -gt 0 ]]; then
  CMAKE_CMD+=( "${CMAKE_EXTRA_ARGS[@]}" )
fi

log "Configuring with CMake..."
"${CMAKE_CMD[@]}"

if [[ ${CONFIGURE_ONLY} -eq 1 ]]; then
  log "Configure-only mode enabled. Skipping build step."
  exit 0
fi

# -------- build --------
log "Building target..."
BUILD_CMD=(cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_JOBS}")
if [[ ${VERBOSE} -eq 1 ]]; then
  BUILD_CMD+=( --verbose )
fi
"${BUILD_CMD[@]}"

# -------- result --------
PLUGIN_PATH="${BUILD_DIR}/libmusa_plugin.so"
if [[ -f "${PLUGIN_PATH}" ]]; then
  log "Build succeeded. Plugin: ${PLUGIN_PATH}"
else
  warn "Build completed, but expected artifact not found at: ${PLUGIN_PATH}"
  warn "Check CMake target name/output path in CMakeLists.txt."
fi