#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing qualiax and all dependencies..."

# Install ffmpeg if not present (required for MP3/AAC/M4A support via pydub)
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found — installing..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y ffmpeg
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg
    else
        echo "Warning: could not install ffmpeg automatically. Install it manually for MP3/AAC support."
    fi
else
    echo "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
fi

# Install qualiax with all Python extras
pip install -e "$SCRIPT_DIR/[all]"

# Optional: install PyTorch for GPU acceleration (CUDA or Apple MPS)
# Uncomment the line for your platform:
# pip install torch                                   # CPU / auto-detect
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

echo ""
echo "Done. Run: qualiax --help"
echo "Try:       qualiax sample.wav"
