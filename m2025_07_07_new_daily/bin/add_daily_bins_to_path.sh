#!/bin/bash
# Add all daily/*/bin directories to PATH if they're not already there
# Source this from your .bashrc: source /Users/dbieber/code/github/dbieber/daily/m2025_07_07_new_daily/bin/add_daily_bins_to_path.sh

DAILY_REPO_PATH="/Users/dbieber/code/github/dbieber/daily"

# Find all bin directories in daily subdirectories
for bin_dir in "$DAILY_REPO_PATH"/m*/bin; do
    # Check if the directory exists and is a directory
    if [[ -d "$bin_dir" ]]; then
        # Check if this path is already in PATH
        if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
            export PATH="$bin_dir:$PATH"
        fi
    fi
done