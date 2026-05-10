#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
USE_GLOBAL=0
INSTALL_TORCH=0
INSTALL_CREPE=0
EXTRAS=("all")

usage() {
    cat <<'EOF'
qualiax installer

Usage:
  ./install.sh [options]

Options:
  --minimal           Install only the base package (no extras).
  --audio             Install the `audio` extra.
  --perceptual        Install the `perceptual` extra.
  --live              Install the `live` extra.
  --watch             Install the `watch` extra.
  --ml                Install the `ml` extra.
  --all               Install the `all` extra (default).
  --with-torch        Also install `torch` and `torchaudio`.
  --with-crepe        Also install `crepe` manually for the neural F0 backend.
  --global            Install into the current Python environment instead of a venv.
  --venv PATH         Virtualenv path to create/use (default: ./.venv).
  --python PATH       Python executable to use (default: python3).
  --help              Show this help.

Examples:
  ./install.sh
  ./install.sh --audio --perceptual
  ./install.sh --minimal --with-torch
  ./install.sh --all --with-torch --with-crepe
EOF
}

set_extras() {
    EXTRAS=("$@")
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --minimal)
            set_extras
            ;;
        --audio)
            if [[ "${EXTRAS[*]}" == "all" ]]; then
                set_extras audio
            else
                EXTRAS+=("audio")
            fi
            ;;
        --perceptual)
            if [[ "${EXTRAS[*]}" == "all" ]]; then
                set_extras perceptual
            else
                EXTRAS+=("perceptual")
            fi
            ;;
        --live)
            if [[ "${EXTRAS[*]}" == "all" ]]; then
                set_extras live
            else
                EXTRAS+=("live")
            fi
            ;;
        --watch)
            if [[ "${EXTRAS[*]}" == "all" ]]; then
                set_extras watch
            else
                EXTRAS+=("watch")
            fi
            ;;
        --ml)
            if [[ "${EXTRAS[*]}" == "all" ]]; then
                set_extras ml
            else
                EXTRAS+=("ml")
            fi
            ;;
        --all)
            set_extras all
            ;;
        --with-torch)
            INSTALL_TORCH=1
            ;;
        --with-crepe)
            INSTALL_CREPE=1
            ;;
        --global)
            USE_GLOBAL=1
            ;;
        --venv)
            shift
            VENV_DIR="${1:?missing value for --venv}"
            ;;
        --python)
            shift
            PYTHON_BIN="${1:?missing value for --python}"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ${#EXTRAS[@]} -gt 1 ]]; then
    declare -A SEEN=()
    UNIQUE_EXTRAS=()
    for extra in "${EXTRAS[@]}"; do
        if [[ -z "${SEEN[$extra]:-}" ]]; then
            SEEN[$extra]=1
            UNIQUE_EXTRAS+=("$extra")
        fi
    done
    EXTRAS=("${UNIQUE_EXTRAS[@]}")
fi

PACKAGE_ARGS=(-e "$SCRIPT_DIR")
if [[ ${#EXTRAS[@]} -gt 0 ]]; then
    IFS=,
    PACKAGE_ARGS=(-e "$SCRIPT_DIR[${EXTRAS[*]}]")
    unset IFS
fi

echo "=================================================="
echo "  qualiax installer"
echo "=================================================="
echo ""
echo "Python: $PYTHON_BIN"

if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "ffmpeg: not found"
    echo "  Install manually if you need MP3/AAC/video-container decoding."
    echo "  macOS:  brew install ffmpeg"
    echo "  Debian: sudo apt install ffmpeg"
fi

if [[ $USE_GLOBAL -eq 1 ]]; then
    PYTHON_EXE="$PYTHON_BIN"
    echo "Target: current Python environment"
else
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        PYTHON_EXE="$VIRTUAL_ENV/bin/python"
        echo "Target: active virtualenv at $VIRTUAL_ENV"
    else
        if [[ ! -d "$VENV_DIR" ]]; then
            echo "Creating virtualenv at $VENV_DIR"
            "$PYTHON_BIN" -m venv "$VENV_DIR"
        fi
        PYTHON_EXE="$VENV_DIR/bin/python"
        echo "Target: virtualenv at $VENV_DIR"
    fi
fi

echo ""
echo "Installing qualiax from pyproject extras..."
"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel
"$PYTHON_EXE" -m pip install "${PACKAGE_ARGS[@]}"

if [[ $INSTALL_TORCH -eq 1 ]]; then
    echo ""
    echo "Installing PyTorch support..."
    "$PYTHON_EXE" -m pip install torch torchaudio
fi

if [[ $INSTALL_CREPE -eq 1 ]]; then
    echo ""
    echo "Installing CREPE (manual opt-in backend)..."
    "$PYTHON_EXE" -m pip install crepe
fi

echo ""
echo "Verifying installation..."
"$PYTHON_EXE" - <<'PY'
import importlib
import sys

mods = [
    ("qualiax", "qualiax"),
    ("click", "click"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
]

failed = []
for mod, label in mods:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "?")
        print(f"  ok  {label} {version}")
    except Exception as exc:
        failed.append((label, exc))

if failed:
    print("")
    for label, exc in failed:
        print(f"  fail {label}: {exc}")
    sys.exit(1)
PY

echo ""
echo "Installation complete."
if [[ $USE_GLOBAL -eq 0 && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the environment with:"
    echo "  source \"$VENV_DIR/bin/activate\""
fi
echo "Try:"
echo "  qualiax --help"
