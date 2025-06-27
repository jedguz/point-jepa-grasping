#!/bin/bash

# Usage:
# ./data/dlrhand2_dataset_loader.sh gs://adlr2025-pointclouds/grasps/student_grasps_v1/studentGrasping_v1.tar.gz ./data/ 10 --mode ordered --by categories
# ./data/dlrhand2_dataset_loader.sh gs://adlr2025-pointclouds/grasps/student_grasps_v1/studentGrasping_v1.tar.gz /mnt/disks/ssd 10 --mode ordered --by categories
set -euo pipefail

# ---- Required arguments
GCS_PATH="$1"
DEST_DIR="$2"
PERCENT="$3"
MODE="random"
BY="categories"

# ---- Optional arguments
shift 3
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --by)
      BY="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ---- Setup
FILENAME=$(basename "$GCS_PATH")
LOCAL_TAR="$DEST_DIR/$FILENAME"
TMP_LIST="$DEST_DIR/all_files.txt"
EXTRACT_LIST="$DEST_DIR/files_to_extract.txt"
mkdir -p "$DEST_DIR"

# ---- Download .tar.gz
echo "📥 Downloading from $GCS_PATH ..."
gsutil cp "$GCS_PATH" "$LOCAL_TAR"

# ---- List files from archive
echo "📃 Indexing archive contents ..."
tar -tzf "$LOCAL_TAR" | grep "^studentGrasping/student_grasps_v1/" > "$TMP_LIST"

# ---- Filter only .obj and .npz
FILTERED_FILES=$(grep -E "\.(obj|npz)$" "$TMP_LIST")

# ---- Build category/model lists
if [[ "$BY" == "categories" ]]; then
  ITEMS=$(echo "$FILTERED_FILES" | cut -d/ -f3 | sort -u)
elif [[ "$BY" == "models" ]]; then
  ITEMS=$(echo "$FILTERED_FILES" | cut -d/ -f3-4 | sort -u)
else
  echo "Invalid --by option: $BY"
  exit 1
fi

TOTAL=$(echo "$ITEMS" | wc -l)
SELECT_NUM=$(( TOTAL * PERCENT / 100 ))

echo "📊 Found $TOTAL $BY. Selecting $SELECT_NUM ($PERCENT%)..."

if [[ "$MODE" == "random" ]]; then
  SELECTED=$(echo "$ITEMS" | shuf | head -n "$SELECT_NUM")
else
  SELECTED=$(echo "$ITEMS" | head -n "$SELECT_NUM")
fi

echo "$SELECTED" > "$DEST_DIR/selected_items.txt"

# ---- Build extraction list
> "$EXTRACT_LIST"
while IFS= read -r ITEM; do
  if [[ "$BY" == "categories" ]]; then
    grep -E "^studentGrasping/student_grasps_v1/${ITEM}/.*/.*\.(obj|npz)$" "$TMP_LIST" >> "$EXTRACT_LIST"
  else
    grep -E "^studentGrasping/student_grasps_v1/${ITEM}/.*\.(obj|npz)$" "$TMP_LIST" >> "$EXTRACT_LIST"
  fi
done < "$DEST_DIR/selected_items.txt"

NUM_FILES=$(wc -l < "$EXTRACT_LIST")
echo "📦 Preparing to extract $NUM_FILES files..."

# ---- Clean previous extracted files (optional)
echo "🧹 Cleaning previous extracted files..."
rm -rf "$DEST_DIR/studentGrasping/student_grasps_v1"

# ---- Extract selected files only
tar -xzf "$LOCAL_TAR" -C "$DEST_DIR" -T "$EXTRACT_LIST"

echo "🧹 Cleaning up temporary files..."
rm -f "$LOCAL_TAR" "$TMP_LIST" "$EXTRACT_LIST"

echo "✅ Done! Extracted $NUM_FILES files to $DEST_DIR"
