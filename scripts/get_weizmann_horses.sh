#!/usr/bin/env bash
# Fetch the 32x32 Weizmann horse dataset (real images + binary GT masks).
#
# Source: the `dvn-horse` release bundles the single-scale Weizmann horses
# pre-resized to 32x32 as `pics32.zip` (images/*.bmp + musks/*.bmp).
# We extract it into datasets/weizmann_horse_32/ with images/ and musks/.
#
# Usage:  bash scripts/get_weizmann_horses.sh [DEST_DIR]
set -euo pipefail

DEST="${1:-datasets/weizmann_horse_32}"
TMP="$(mktemp -d)"
REPO="https://github.com/nkg114mc/dvn-horse.git"

echo "Cloning $REPO ..."
git clone --depth 1 "$REPO" "$TMP/dvn-horse"

echo "Extracting pics32.zip ..."
unzip -q -o "$TMP/dvn-horse/pics32.zip" -d "$TMP/out"

mkdir -p "$DEST"
cp -r "$TMP/out/pics32/." "$DEST/"
rm -rf "$TMP"

n_img=$(find "$DEST/images" -name '*.bmp' | wc -l | tr -d ' ')
n_msk=$(find "$DEST/musks" -name '*.bmp' | wc -l | tr -d ' ')
echo "Done: $n_img images, $n_msk masks in $DEST"
echo "Run: PYTHONPATH=src python experiments/phase2_real_horses.py --data $DEST"
