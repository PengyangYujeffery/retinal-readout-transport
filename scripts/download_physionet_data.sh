#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal, interactive downloader for the BRSET/mBRSET embedding study.
# Run this script manually on MeluXina. It deliberately does not accept a
# password through arguments or environment variables.

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v wget >/dev/null 2>&1 || die "wget is required"

data_root="${AICS_DATA_ROOT:-}"
[[ -n "$data_root" ]] || die "Set AICS_DATA_ROOT to a protected MeluXina project path"
[[ "$data_root" = /* ]] || die "AICS_DATA_ROOT must be an absolute path"

case "$data_root" in
  /project/home/*|/project/scratch/*) ;;
  *) die "AICS_DATA_ROOT must be under /project/home or /project/scratch" ;;
esac

umask 077
raw_dir="$data_root/raw"
mkdir -p "$raw_dir"
chmod 700 "$data_root" "$raw_dir"

if [[ -z "${PHYSIONET_USER:-}" ]]; then
  read -r -p 'PhysioNet username: ' PHYSIONET_USER
fi
[[ -n "$PHYSIONET_USER" ]] || die "PhysioNet username cannot be empty"

base="https://physionet.org/files"
urls=(
  "$base/embedding-brset-mbrset/1.0.0/Embeddings_brset_dinov3_vits16.csv"
  "$base/embedding-brset-mbrset/1.0.0/Embeddings_mbrset_dinov3_vits16.csv"
  "$base/embedding-brset-mbrset/1.0.0/Embeddings_brset_dinov3_convnext_tiny.csv"
  "$base/embedding-brset-mbrset/1.0.0/Embeddings_mbrset_dinov3_convnext_tiny.csv"
  "$base/brazilian-ophthalmological/1.0.1/"
  "$base/mbrset/1.0/labels_mbrset.csv"
)

printf 'Downloading four embeddings and two metadata tables to %s\n' "$raw_dir"
printf 'wget will now prompt once for the PhysioNet password.\n'

url_manifest="$(mktemp)"
trap 'rm -f "$url_manifest"' EXIT
printf '%s\n' "${urls[@]}" > "$url_manifest"

wget \
  --recursive \
  --no-parent \
  --no-directories \
  --level=3 \
  --accept='*.csv' \
  --reject='index.html*' \
  --continue \
  --tries=5 \
  --waitretry=5 \
  --user="$PHYSIONET_USER" \
  --ask-password \
  --directory-prefix="$raw_dir" \
  --input-file="$url_manifest"

find "$raw_dir" -maxdepth 1 -type f -name '*.csv' -exec chmod 600 {} +
printf 'Download complete. Run verify_data.py before analysis.\n'
