#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_patch() {
  local repo_dir="$1"
  local patch_file="$2"

  if [[ ! -d "$repo_dir" ]]; then
    echo "Missing repo dir: $repo_dir" >&2
    return 1
  fi
  if [[ ! -f "$patch_file" ]]; then
    echo "Missing patch file: $patch_file" >&2
    return 1
  fi

  if git -C "$repo_dir" apply --ignore-whitespace --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Already applied: $(basename "$patch_file")"
    return 0
  fi

  git -C "$repo_dir" apply --ignore-whitespace --check "$patch_file"
  git -C "$repo_dir" apply --ignore-whitespace "$patch_file"
  echo "Applied: $(basename "$patch_file")"
}

git -C "$ROOT" submodule update --init --recursive

apply_patch "$ROOT/3rdparty/gDel3D" "$ROOT/patches/gDel3D_cuda13_clockrate.patch"
apply_patch "$ROOT/3rdparty/pytorch3d" "$ROOT/3rdparty/pytorch3d.patch"
apply_patch "$ROOT/3rdparty/kaolin" "$ROOT/3rdparty/kaolin.patch"
apply_patch "$ROOT/3rdparty/open3d" "$ROOT/3rdparty/open3d.patch"
