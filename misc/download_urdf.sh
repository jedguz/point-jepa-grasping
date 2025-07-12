#!/bin/bash

# Variables
DEST_DIR="./newurdf"
LOCAL_TAR="$DEST_DIR/studentGrasping_v1.tar.gz"
GCS_PATH="gs://adlr2025-pointclouds/grasps/student_grasps_v1/studentGrasping_v1.tar.gz"

# Create destination directory
mkdir -p "$DEST_DIR"

# Download the .tar.gz file to local path
echo "Downloading from GCS..."
gsutil cp "$GCS_PATH" "$LOCAL_TAR" || { echo "Download failed"; exit 1; }

# Extract only the urdfs/ folder
echo "Extracting URDFs..."
tar -xzf "$LOCAL_TAR" -C "$DEST_DIR" --wildcards 'studentGrasping/urdfs/*' --strip-components=1

echo "Extraction complete. URDFs are in: $DEST_DIR/urdfs"
