#!/bin/bash

# Usage:
# ./data/dlrhand2_dataset_loader.sh gs://adlr2025-pointclouds/grasps/student_grasps_v1/studentGrasping_v1.tar.gz ./data/ 10 --mode ordered --by models
#
# Choose the percent of the dataset to be downloaded: 10|25|50|75|100
#
# Defaults:
#   --mode random (choose random or ordered)
#   --by categories (choose models or categories)

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <gs://bucket/file.tar.gz> <destination-dir> <percent> [--mode random|ordered] [--by categories|models]"
    exit 1
fi

GCS_PATH="$1"
DEST_DIR="$2"
PERCENT="$3"

# Defaults
MODE="random"
BY="categories"

# Parse optional args
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

# Validate percent
if ! [[ "$PERCENT" =~ ^(10|25|50|75|100)$ ]]; then
    echo "❌ Invalid percent: $PERCENT. Must be one of: 10, 25, 50, 75, 100"
    exit 1
fi

# Validate mode
if ! [[ "$MODE" =~ ^(random|ordered)$ ]]; then
    echo "❌ Invalid mode: $MODE. Must be 'random' or 'ordered'"
    exit 1
fi

# Validate BY
if ! [[ "$BY" =~ ^(categories|models)$ ]]; then
    echo "❌ Invalid by: $BY. Must be 'categories' or 'models'"
    exit 1
fi

FILENAME=$(basename "$GCS_PATH")
LOCAL_TAR="$DEST_DIR/$FILENAME"

mkdir -p "$DEST_DIR"

echo "📥 Downloading archive from $GCS_PATH ..."
gsutil cp "$GCS_PATH" "$LOCAL_TAR"

echo "📃 Listing archive contents..."
tar -tzf "$LOCAL_TAR" > "$DEST_DIR/all_files.txt"

echo "🔧 Sampling mode: $MODE"
echo "🔧 Sampling by: $BY"

# Filter only files inside student_grasps_v1
V1_PATHS=$(grep "^studentGrasping/student_grasps_v1/" "$DEST_DIR/all_files.txt")

if [[ "$BY" == "categories" ]]; then
    # Extract unique categories (3rd folder)
    CATEGORIES=$(echo "$V1_PATHS" | cut -d/ -f3 | sort -u | grep -v '^$')
    TOTAL_CATEGORIES=$(echo "$CATEGORIES" | wc -l)
    NUM_TO_SELECT=$(( TOTAL_CATEGORIES * PERCENT / 100 ))

    echo "📊 Found $TOTAL_CATEGORIES categories. Selecting $NUM_TO_SELECT (~$PERCENT%)"

    if [[ "$MODE" == "random" ]]; then
        SELECTED_CATEGORIES=$(echo "$CATEGORIES" | shuf | head -n "$NUM_TO_SELECT")
    else
        SELECTED_CATEGORIES=$(echo "$CATEGORIES" | head -n "$NUM_TO_SELECT")
    fi

    echo "$SELECTED_CATEGORIES" > "$DEST_DIR/selected_categories.txt"

    > "$DEST_DIR/files_to_extract.txt"
    while read -r CATEGORY; do
        echo "$V1_PATHS" | grep -E "^studentGrasping/student_grasps_v1/${CATEGORY}/.*\.(obj|npz)$" >> "$DEST_DIR/files_to_extract.txt"
    done <<< "$SELECTED_CATEGORIES"

elif [[ "$BY" == "models" ]]; then
    # Extract unique category/model pairs (3rd and 4th folder)
    MODELS=$(echo "$V1_PATHS" | awk -F/ '{print $3 "/" $4}' | sort -u)
    TOTAL_MODELS=$(echo "$MODELS" | wc -l)
    NUM_TO_SELECT=$(( TOTAL_MODELS * PERCENT / 100 ))

    echo "📊 Found $TOTAL_MODELS models. Selecting $NUM_TO_SELECT (~$PERCENT%)"

    if [[ "$MODE" == "random" ]]; then
        SELECTED_MODELS=$(echo "$MODELS" | shuf | head -n "$NUM_TO_SELECT")
    else
        SELECTED_MODELS=$(echo "$MODELS" | head -n "$NUM_TO_SELECT")
    fi

    echo "$SELECTED_MODELS" > "$DEST_DIR/selected_models.txt"

    > "$DEST_DIR/files_to_extract.txt"
    while read -r MODEL; do
        echo "$V1_PATHS" | grep -E "^studentGrasping/student_grasps_v1/${MODEL}/.*\.(obj|npz)$" >> "$DEST_DIR/files_to_extract.txt"
    done <<< "$SELECTED_MODELS"

else
    echo "❌ Invalid --by value. Must be 'categories' or 'models'"
    exit 1
fi

NUM_FILES=$(wc -l < "$DEST_DIR/files_to_extract.txt")
echo "📦 Extracting $NUM_FILES files from selected items..."

tar -xzf "$LOCAL_TAR" -C "$DEST_DIR" -T "$DEST_DIR/files_to_extract.txt"

echo "✅ Extraction complete! Files extracted to: $DEST_DIR"
