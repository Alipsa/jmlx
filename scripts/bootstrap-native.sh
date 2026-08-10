#!/usr/bin/env bash
#
# Downloads, builds, and stages everything jmlx needs to talk to real MLX on
# Apple Silicon: jextract, the mlx-metal wheel (headers + prebuilt dylibs +
# metallib), and mlx-c (built from source against the wheel's MLX). Idempotent
# -- safe to re-run. See req/initial-plan.md §3 for the full rationale; this
# script is that section made executable.
#
# Prerequisites this script assumes are already installed (developer-machine
# setup, not something a project script should silently do): cmake, git,
# curl, unzip, otool, codesign. jextract itself IS downloaded by this script.
#
# Usage: ./scripts/bootstrap-native.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_DIR="$REPO_ROOT/native"
SCRATCH_DIR="$NATIVE_DIR/scratch"
INSTALL_DIR="$NATIVE_DIR/install"

# --- Pinned versions ---------------------------------------------------------
# req/initial-plan.md Decision 9: this mlx-c commit and this mlx-metal wheel
# version both track MLX v0.31.2 exactly, sidestepping the wheel/mlx-c version
# mismatch (latest wheel is 0.32.0; mlx-c's CMakeLists.txt pins v0.31.2)
# instead of gambling on newer headers compiling against older mlx-c source.
MLX_METAL_VERSION="0.31.2"
MLX_METAL_WHEEL_URL="https://files.pythonhosted.org/packages/99/82/11fd62a8d7a3e96e5c43220b17de0151e3f10101f8bb3b865f5bd9cdd074/mlx_metal-${MLX_METAL_VERSION}-py3-none-macosx_26_0_arm64.whl"
MLX_METAL_WHEEL_SHA256="84ffb60ee503f03eb684f5fb168d5cff31e2a16b7f27c1731eaf7662bd6e9b46"

MLX_C_REPO="https://github.com/ml-explore/mlx-c.git"
MLX_C_COMMIT="fba4470b89073180056c9ea46c443051375f7399"

JEXTRACT_PAGE="https://jdk.java.net/jextract/"
# Trust-on-first-use: jextract's own .sha256 comes from the same origin as the
# artifact, so verifying against it proves the download wasn't corrupted, not
# that it's authentic. Instead, record the sha256 resolved on the first
# successful run and compare on every later run.
JEXTRACT_PIN_FILE="$REPO_ROOT/scripts/.jextract-sha256-pin"

# Real measured size (req/initial-plan.md) is 157,748,008 bytes; use a
# generous floor so a minor MLX version bump doesn't false-positive here.
METALLIB_MIN_BYTES=100000000

log() { printf '>>> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for tool in cmake git curl unzip otool codesign shasum; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool '$tool' not found on PATH -- install it (e.g. 'brew install $tool') and re-run"
done

mkdir -p "$SCRATCH_DIR"

# --- 1. jextract --------------------------------------------------------------
JEXTRACT_DIR="$SCRATCH_DIR/jextract"
JEXTRACT_BIN="$JEXTRACT_DIR/bin/jextract"

if [[ -x "$JEXTRACT_BIN" ]]; then
  log "jextract already staged at $JEXTRACT_BIN"
else
  log "Resolving jextract build for macos-aarch64 from $JEXTRACT_PAGE"
  JEXTRACT_URL="$(curl -sL -A 'Mozilla/5.0' "$JEXTRACT_PAGE" \
    | grep -oE 'https://download\.java\.net/java/early_access/jextract/[^"]*macos-aarch64[^"]*\.tar\.gz' \
    | sort -u | head -1)"
  [[ -n "$JEXTRACT_URL" ]] || die "could not resolve a macos-aarch64 jextract download URL from $JEXTRACT_PAGE"
  log "Resolved jextract download: $JEXTRACT_URL"

  JEXTRACT_TARBALL="$SCRATCH_DIR/jextract.tar.gz"
  curl -sL -o "$JEXTRACT_TARBALL" "$JEXTRACT_URL"
  ACTUAL_SHA256="$(shasum -a 256 "$JEXTRACT_TARBALL" | awk '{print $1}')"

  if [[ -f "$JEXTRACT_PIN_FILE" ]]; then
    PINNED_SHA256="$(cat "$JEXTRACT_PIN_FILE")"
    if [[ "$ACTUAL_SHA256" != "$PINNED_SHA256" ]]; then
      die "jextract download sha256 mismatch: expected $PINNED_SHA256 (pinned in $JEXTRACT_PIN_FILE from an earlier trusted run), got $ACTUAL_SHA256. Refusing to use a build that changed since then."
    fi
  else
    log "No jextract pin recorded yet -- trusting this download and pinning its sha256 ($ACTUAL_SHA256) to $JEXTRACT_PIN_FILE"
    echo "$ACTUAL_SHA256" > "$JEXTRACT_PIN_FILE"
  fi

  rm -rf "$JEXTRACT_DIR"
  mkdir -p "$JEXTRACT_DIR"
  tar xzf "$JEXTRACT_TARBALL" -C "$JEXTRACT_DIR" --strip-components=1
  [[ -x "$JEXTRACT_BIN" ]] || die "jextract archive did not produce an executable at $JEXTRACT_BIN"
fi
"$JEXTRACT_BIN" --version

# --- 2. mlx-metal wheel --------------------------------------------------------
WHEEL_DIR="$SCRATCH_DIR/wheel"
WHEEL_MARKER="$WHEEL_DIR/.version-${MLX_METAL_VERSION}"

if [[ -f "$WHEEL_MARKER" ]]; then
  log "mlx-metal $MLX_METAL_VERSION wheel already unpacked at $WHEEL_DIR"
else
  log "Downloading mlx-metal $MLX_METAL_VERSION (macosx_26_0_arm64)"
  WHEEL_FILE="$SCRATCH_DIR/mlx_metal.whl"
  curl -sL -o "$WHEEL_FILE" "$MLX_METAL_WHEEL_URL"
  ACTUAL_SHA256="$(shasum -a 256 "$WHEEL_FILE" | awk '{print $1}')"
  [[ "$ACTUAL_SHA256" == "$MLX_METAL_WHEEL_SHA256" ]] || die \
    "mlx-metal wheel sha256 mismatch: expected $MLX_METAL_WHEEL_SHA256, got $ACTUAL_SHA256"

  rm -rf "$WHEEL_DIR"
  mkdir -p "$WHEEL_DIR"
  # A wheel is a zip; extract directly -- no pip, no virtualenv.
  unzip -q "$WHEEL_FILE" -d "$WHEEL_DIR"
  touch "$WHEEL_MARKER"
fi

METALLIB_SRC="$WHEEL_DIR/mlx/lib/mlx.metallib"
[[ -f "$METALLIB_SRC" ]] || die "expected $METALLIB_SRC after unpacking the wheel -- has the wheel layout changed?"
METALLIB_SIZE="$(stat -f %z "$METALLIB_SRC")"
[[ "$METALLIB_SIZE" -ge "$METALLIB_MIN_BYTES" ]] || die \
  "mlx.metallib is only $METALLIB_SIZE bytes (expected at least $METALLIB_MIN_BYTES) -- the wheel download looks wrong"
log "mlx.metallib: $METALLIB_SIZE bytes"

# --- 3. mlx-c clone at the pinned commit --------------------------------------
MLXC_DIR="$SCRATCH_DIR/mlx-c"
if [[ -d "$MLXC_DIR/.git" ]]; then
  CURRENT_COMMIT="$(git -C "$MLXC_DIR" rev-parse HEAD)"
  if [[ "$CURRENT_COMMIT" != "$MLX_C_COMMIT" ]]; then
    log "mlx-c is at $CURRENT_COMMIT, fetching and checking out pinned $MLX_C_COMMIT"
    git -C "$MLXC_DIR" fetch origin
    git -C "$MLXC_DIR" checkout --detach "$MLX_C_COMMIT"
  else
    log "mlx-c already at pinned commit $MLX_C_COMMIT"
  fi
else
  log "Cloning mlx-c and checking out pinned commit $MLX_C_COMMIT"
  rm -rf "$MLXC_DIR"
  git clone "$MLX_C_REPO" "$MLXC_DIR"
  git -C "$MLXC_DIR" checkout --detach "$MLX_C_COMMIT"
fi

# --- 4. Assert the version pairing before building ----------------------------
# find_package(MLX REQUIRED) carries no version constraint of its own -- this
# check stands in for it.
MLXC_PINNED_MLX_TAG="$(grep -oE 'GIT_TAG v[0-9.]+' "$MLXC_DIR/CMakeLists.txt" | awk '{print $2}')"
[[ -n "$MLXC_PINNED_MLX_TAG" ]] || die "could not find mlx-c's FetchContent GIT_TAG in its CMakeLists.txt -- has the file moved or been restructured?"
EXPECTED_TAG="v${MLX_METAL_VERSION}"
if [[ "$MLXC_PINNED_MLX_TAG" != "$EXPECTED_TAG" ]]; then
  log "mlx-c at $MLX_C_COMMIT is written against MLX $MLXC_PINNED_MLX_TAG, but this script pins mlx-metal $MLX_METAL_VERSION."
  log "This pairing was chosen deliberately once (req/initial-plan.md Decision 9); a mismatch here means that decision needs revisiting, not silently overriding."
  die "version pairing assertion failed: mlx-c wants $MLXC_PINNED_MLX_TAG, wheel is $EXPECTED_TAG"
fi
log "Version pairing OK: mlx-c $MLX_C_COMMIT <-> mlx-metal $MLX_METAL_VERSION (both track $EXPECTED_TAG)"

# --- 5. Configure + build mlx-c against the wheel -----------------------------
BUILD_DIR="$MLXC_DIR/build"
CONFIGURE_LOG="$SCRATCH_DIR/cmake-configure.log"
log "Configuring mlx-c (MLX_C_USE_SYSTEM_MLX=ON, BUILD_SHARED_LIBS=ON)"
# --fresh discards any cached CMakeCache.txt from a prior run of this script.
# Without it, a cached "Found MLX" result isn't re-printed on reconfigure,
# which would make the fast-path assertion below fail on every run after the
# first even though the fast path is (still) in effect.
cmake --fresh -S "$MLXC_DIR" -B "$BUILD_DIR" \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DMLX_DIR="$WHEEL_DIR/mlx/share/cmake/MLX" \
  -DCMAKE_PREFIX_PATH="$WHEEL_DIR/mlx" \
  -DCMAKE_INSTALL_RPATH=@loader_path \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  | tee "$CONFIGURE_LOG"

# Assert the fast path actually engaged. If MLX_C_USE_SYSTEM_MLX=ON silently
# failed to take, mlx-c still builds -- just against a different, FetchContent'd
# MLX -- and nothing else here would catch it.
if [[ -d "$BUILD_DIR/_deps/mlx-src" ]]; then
  die "$BUILD_DIR/_deps/mlx-src exists -- MLX_C_USE_SYSTEM_MLX=ON did not take effect; mlx-c FetchContent'd its own MLX instead of using the wheel's"
fi
grep -qi "Found MLX" "$CONFIGURE_LOG" || die "cmake configure log does not report finding MLX under the wheel prefix"
log "Fast path confirmed: no _deps/mlx-src, MLX found under the wheel prefix"

log "Building mlxc"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)"

log "Installing mlxc headers + library to $INSTALL_DIR"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

# --- 6. Assemble the flat runtime directory -----------------------------------
# Runtime invariant (req/initial-plan.md): all four artifacts live in one flat
# directory. MLX finds mlx.metallib by colocation with its own image, and dyld
# resolves the @rpath siblings the same way.
RUNTIME_LIB_DIR="$INSTALL_DIR/lib"
mkdir -p "$RUNTIME_LIB_DIR"
cp "$WHEEL_DIR/mlx/lib/libmlx.dylib" "$WHEEL_DIR/mlx/lib/libjaccl.dylib" "$METALLIB_SRC" "$RUNTIME_LIB_DIR/"

# --- 7. Assert the result, don't trust it -------------------------------------
log "Verifying libmlxc.dylib's dependencies resolve against the flat directory"
if otool -L "$RUNTIME_LIB_DIR/libmlxc.dylib" | grep -q "not found"; then
  die "otool -L reports an unresolved dependency for libmlxc.dylib -- the flat runtime directory is not self-consistent"
fi

for lib in libmlxc.dylib libmlx.dylib libjaccl.dylib; do
  codesign -v "$RUNTIME_LIB_DIR/$lib" || die "$lib failed codesign verification (a post-link install_name_tool edit without re-signing would show up here)"
done

STAGED_METALLIB_SIZE="$(stat -f %z "$RUNTIME_LIB_DIR/mlx.metallib")"
[[ "$STAGED_METALLIB_SIZE" -ge "$METALLIB_MIN_BYTES" ]] || die \
  "staged mlx.metallib is only $STAGED_METALLIB_SIZE bytes -- expected at least $METALLIB_MIN_BYTES"

log "native/install/lib:"
ls -la "$RUNTIME_LIB_DIR"

log "Bootstrap complete."
log "Next: ./scripts/regen-bindings.sh to (re)generate jmlx-ffi's jextract bindings."
