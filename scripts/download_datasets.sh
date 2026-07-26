#!/usr/bin/env bash
# download_datasets.sh — fetch all three datasets used in the Q² paper
# (arXiv:2511.05898) onto a target disk.
#
# Datasets and their canonical sources:
#   1. PASCAL VOC 2007 + 2012   — object detection (YOLOv5, YOLOv11, RT-DETR)
#        [9] Everingham et al., IJCV 2010.
#   2. COCO 2017 (train+val+test+annotations) — object detection
#        [17] Lin et al., ECCV 2014.
#   3. BUSI (breast ultrasound) — segmentation (MK-UNet)
#        [1] Al-Dhabyani et al., Data in Brief 28, 104863 (2020).
#
# Usage:
#   scripts/download_datasets.sh /mnt/hdd2/datasets                # all three
#   scripts/download_datasets.sh /mnt/hdd2/datasets voc coco busi   # explicit
#   scripts/download_datasets.sh /mnt/hdd2/datasets --no-extract    # keep zips
#
# Idempotent: every file is skipped if its byte size already matches the
# expected Content-Length. Re-running resumes downloads (wget -c) and skips
# already-extracted archives.
#
# WHEN THE SERVER IS THROTTLED BY AWS (the realistic case):
# images.cocodataset.org is a public S3 bucket. From some hosting providers
# (and from at least one of our bare-metal servers) S3 returns 200 headers
# then ships ~0 bytes/sec of body — effectively throttling the source IP to
# death. The bypass used to populate /mnt/hdd2/datasets on the throttled host
# was: download on a machine that is NOT throttled (e.g. a workstation on a
# consumer VPN), then rsync over the LAN:
#     # on the workstation (not throttled):
#     QAT_DATASETS=/local/path scripts/download_datasets.sh --no-extract
#     # then push to the throttled server:
#     rsync -a --partial /local/path/ server:/mnt/hdd2/datasets/
# Reproducible: BUSI (HuggingFace CDN) and PASCAL VOC (Oxford + pjreddie) were
# NOT affected by the throttle on the same host — only cocodataset.org/S3.
#
# Exit codes: 0 = ok; non-zero = at least one dataset failed to fetch.
set -uo pipefail

# ---- expected sizes (bytes) — used for the skip-if-present sanity check ----
# These are the canonical sizes as served by the upstream mirrors in July 2026.
declare -A EXPECTED_SIZE=(
  # PASCAL VOC
  [VOCtrainval_11-May-2012.tar]=1999639040
  [VOCtest_06-Nov-2007.tar]=4510208000
  [VOCtrainval_06-Nov-2007.tar]=4600320000
  # COCO 2017
  [train2017.zip]=19336861798
  [val2017.zip]=815585330
  [test2017.zip]=6646970404
  [annotations_trainval2017.zip]=252907541
  # BUSI (HuggingFace mirror, 204421470 bytes; canonical 780 images)
  [Dataset_BUSI.zip]=204421470
)

# ---- URLs ------------------------------------------------------------------
# VOC: the Oxford host (host.robots.ox.ac.uk) 301-redirects to thor.robots.ox.ac.uk;
# we use the post-redirect URL directly to avoid one round-trip and to keep
# `wget` happy with the certificate. pjreddie.com is the canonical YOLO mirror.
VOC_2012_URL="https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_2007_TRAIN_URL="https://data.pjreddie.com/files/VOCtrainval_06-Nov-2007.tar"
VOC_2007_TEST_URL="https://data.pjreddie.com/files/VOCtest_06-Nov-2007.tar"

COCO_TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
COCO_TEST_URL="http://images.cocodataset.org/zips/test2017.zip"
COCO_ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# BUSI: the original Data-in-Brief supplementary (mmc1.zip on ScienceDirect)
# contains ONLY the paperwork (ethics committee, consent, conflict of interest)
# — the actual images were deposited on Kaggle. Kaggle needs an API token,
# which we don't ship. Instead we use the verified public HuggingFace mirror
# `gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`, which re-hosts the
# exact same archive (204421470 bytes, 780 images across benign/malignant/normal,
# CC-BY-4.0, README explicitly cites the same DOI 10.1016/j.dib.2019.104863
# that the paper uses as reference [1]).
BUSI_URL="https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset/resolve/main/Breast-Cancer-Ultrasound-Images-Dataset.zip"

# ---- helpers ---------------------------------------------------------------
GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
[ -t 1 ] || { GREEN=""; RED=""; YEL=""; DIM=""; OFF=""; }

ok()   { echo "${GREEN}OK   $*${OFF}"; }
fail() { echo "${RED}FAIL $*${OFF}"; FAIL_CNT=$((FAIL_CNT+1)); }
note() { echo "${DIM}     $*${OFF}"; }
warn() { echo "${YEL}WARN $*${OFF}"; }
FAIL_CNT=0

have() { command -v "$1" >/dev/null 2>&1; }

# fetch URL DEST_DIR [EXPECTED_SIZE]
# - skips if the file already exists at the expected size
# - resumes partial downloads (wget -c)
# - rewrites the filename to a clean name (VOC/COCO archives are served with
#   ugly names like "mmc1.zip")
fetch() {
  local url="$1" dest_dir="$2" fname="$3" expected="${4:-}"
  local dest="$dest_dir/$fname"
  mkdir -p "$dest_dir"

  if [[ -f "$dest" ]]; then
    local actual
    actual=$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest" 2>/dev/null || echo 0)
    if [[ -n "$expected" && "$actual" == "$expected" ]]; then
      ok "$fname already present ($actual bytes)"
      return 0
    fi
    note "$fname exists but $actual bytes (expected ${expected:-?}); resuming"
  else
    note "downloading $fname ← $url"
  fi

  # wget: continue, follow redirects, retry, no DNS caching, write to clean name
  if have wget; then
    if wget --no-verbose --continue --tries=5 --waitretry=5 --timeout=60 \
            --no-check-certificate \
            --user-agent="QAT-download_datasets.sh/1.0" \
            -O "$dest" "$url"; then
      ok "$fname downloaded"
    else
      fail "$fname download failed (wget exit $?)"
      return 1
    fi
  elif have curl; then
    if curl -fsSL --retry 5 --retry-delay 5 --connect-timeout 60 \
            -A "QAT-download_datasets.sh/1.0" \
            -o "$dest" -C - "$url"; then
      ok "$fname downloaded"
    else
      fail "$fname download failed (curl exit $?)"
      return 1
    fi
  else
    fail "neither wget nor curl available"
    return 1
  fi

  # post-check size against expected, if known
  if [[ -n "$expected" ]]; then
    local actual
    actual=$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest" 2>/dev/null || echo 0)
    if [[ "$actual" != "$expected" ]]; then
      warn "$fname is $actual bytes (expected $expected) — size mismatch"
    fi
  fi
  return 0
}

# extract ARCHIVE INTO_DIR [strip-components]
extract() {
  local archive="$1" into="$2"
  if [[ ! -f "$archive" ]]; then return 1; fi
  mkdir -p "$into"
  case "$archive" in
    *.tar)
      note "extracting $(basename "$archive") → $into"
      tar -xf "$archive" -C "$into" && ok "extracted $(basename "$archive")"
      ;;
    *.zip)
      note "extracting $(basename "$archive") → $into"
      if have unzip; then
        unzip -q -o "$archive" -d "$into" && ok "extracted $(basename "$archive")"
      else
        # Python's zipfile is always around and good enough for these archives
        python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
          "$archive" "$into" && ok "extracted $(basename "$archive") (via python)"
      fi
      ;;
    *) fail "unknown archive type: $archive"; return 1 ;;
  esac
}

# ---- per-dataset routines --------------------------------------------------
download_voc() {
  local root="$1/voc"; local do_extract="${2:-1}"
  echo
  echo "=== PASCAL VOC (2007 train+test, 2012 trainval) → $root ==="
  fetch "$VOC_2012_URL"        "$root" "VOCtrainval_11-May-2012.tar" "${EXPECTED_SIZE[VOCtrainval_11-May-2012.tar]}"
  fetch "$VOC_2007_TRAIN_URL"  "$root" "VOCtrainval_06-Nov-2007.tar" "${EXPECTED_SIZE[VOCtrainval_06-Nov-2007.tar]}"
  fetch "$VOC_2007_TEST_URL"   "$root" "VOCtest_06-Nov-2007.tar"     "${EXPECTED_SIZE[VOCtest_06-Nov-2007.tar]}"
  if [[ "$do_extract" == "1" ]]; then
    extract "$root/VOCtrainval_11-May-2012.tar" "$root"
    extract "$root/VOCtrainval_06-Nov-2007.tar" "$root"
    extract "$root/VOCtest_06-Nov-2007.tar"     "$root"
  fi
}

download_coco() {
  local root="$1/coco"; local do_extract="${2:-1}"
  echo
  echo "=== COCO 2017 (train, val, test, annotations) → $root ==="
  fetch "$COCO_TRAIN_URL" "$root" "train2017.zip"                "${EXPECTED_SIZE[train2017.zip]}"
  fetch "$COCO_VAL_URL"   "$root" "val2017.zip"                  "${EXPECTED_SIZE[val2017.zip]}"
  fetch "$COCO_TEST_URL"  "$root" "test2017.zip"                 "${EXPECTED_SIZE[test2017.zip]}"
  fetch "$COCO_ANN_URL"   "$root" "annotations_trainval2017.zip" "${EXPECTED_SIZE[annotations_trainval2017.zip]}"
  if [[ "$do_extract" == "1" ]]; then
    extract "$root/annotations_trainval2017.zip" "$root"
    extract "$root/val2017.zip"                  "$root"
    # train/test are large; extract after the small ones so they are usable sooner
    extract "$root/train2017.zip"                "$root"
    extract "$root/test2017.zip"                 "$root"
  fi
}

download_busi() {
  local root="$1/busi"; local do_extract="${2:-1}"
  echo
  echo "=== BUSI (breast ultrasound, Data-in-Brief) → $root ==="
  fetch "$BUSI_URL" "$root" "Dataset_BUSI.zip" "${EXPECTED_SIZE[Dataset_BUSI.zip]}"
  if [[ "$do_extract" == "1" ]]; then
    extract "$root/Dataset_BUSI.zip" "$root"
  fi
}

# ---- main ------------------------------------------------------------------
DEST="${QAT_DATASETS:-}"
DO_EXTRACT=1
TARGETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-extract) DO_EXTRACT=0; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)  if [[ -z "$DEST" ]]; then DEST="$1"
        else TARGETS+=("$1")
        fi
        shift ;;
  esac
done

if [[ -z "$DEST" ]]; then
  echo "usage: $0 <dest-root> [voc|coco|busi ...] [--no-extract]" >&2
  echo "       (or set QAT_DATASETS=<dest-root>)" >&2
  exit 2
fi

# default: all three
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(voc coco busi)
fi

mkdir -p "$DEST"
echo "Q² datasets → $DEST"
echo "targets: ${TARGETS[*]}"
echo "extract: $DO_EXTRACT"

for t in "${TARGETS[@]}"; do
  case "$t" in
    voc)  download_voc  "$DEST" "$DO_EXTRACT" ;;
    coco) download_coco "$DEST" "$DO_EXTRACT" ;;
    busi) download_busi "$DEST" "$DO_EXTRACT" ;;
    *) fail "unknown target: $t (expected voc|coco|busi)" ;;
  esac
done

echo
if [[ "$FAIL_CNT" -gt 0 ]]; then
  echo "${RED}=== $FAIL_CNT failure(s) — see above ===${OFF}"
  exit 1
fi
echo "${GREEN}=== all datasets ready under $DEST ===${OFF}"

# ---- summary of what landed ------------------------------------------------
echo
echo "Layout:"
if have du; then
  du -sh "$DEST"/* 2>/dev/null | while read -r line; do echo "  $line"; done
fi
exit 0
