#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "  qualiax installer"
echo "=================================================="
echo ""

# ── System: ffmpeg ────────────────────────────────────────────────────────────
echo "[1/4] Checking ffmpeg (required for MP3 / AAC / M4A / Opus)..."
if ! command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg not found — installing..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    else
        echo "  !! Could not install ffmpeg automatically."
        echo "     Install it manually: https://ffmpeg.org/download.html"
    fi
else
    echo "  OK: $(ffmpeg -version 2>&1 | head -1)"
fi

# ── Python packages: core + all extras ───────────────────────────────────────
echo ""
echo "[2/4] Installing qualiax + all Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e "$SCRIPT_DIR/[all]"

# ── PyTorch (GPU acceleration) ────────────────────────────────────────────────
echo ""
echo "[3/4] Checking PyTorch (optional — enables CUDA / Apple MPS acceleration)..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    DEVICE="CPU"
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="CUDA ($(python -c "import torch; print(torch.cuda.get_device_name(0))"))"
    elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        DEVICE="MPS (Apple Silicon)"
    fi
    echo "  OK: PyTorch $TORCH_VER — active device: $DEVICE"
else
    echo "  PyTorch not found. Installing CPU build..."
    echo "  (For CUDA or MPS, install manually — see: https://pytorch.org/get-started)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
python -c "
import importlib, sys

ok  = []
err = []
deps = [
    ('click',     'click'),
    ('numpy',     'numpy'),
    ('scipy',     'scipy'),
    ('soundfile', 'soundfile'),
    ('pydub',     'pydub'),
    ('pesq',      'pesq'),
    ('pystoi',    'pystoi'),
    ('librosa',   'librosa'),
    ('torch',     'torch (GPU acceleration)'),
]
for mod, label in deps:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        ok.append(f'  ✓  {label} {ver}')
    except ImportError:
        err.append(f'  ✗  {label}')

print('  Installed:')
for line in ok:  print(line)
if err:
    print()
    print('  Not installed (optional):')
    for line in err: print(line)
"

echo ""
echo "=================================================="
echo "  Installation complete."
echo ""
echo "  Run:  qualiax --help"
echo "  Try:  qualiax sample.wav"
echo "=================================================="
