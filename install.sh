#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
USE_GLOBAL=0
INSTALL_TORCH=0
INSTALL_CREPE=0
RUN_DEMO=0
PROFILE="all"

if [[ -t 1 ]]; then
  BOLD="$(printf '\033[1m')"
  DIM="$(printf '\033[2m')"
  GREEN="$(printf '\033[32m')"
  YELLOW="$(printf '\033[33m')"
  CYAN="$(printf '\033[36m')"
  RESET="$(printf '\033[0m')"
else
  BOLD=""
  DIM=""
  GREEN=""
  YELLOW=""
  CYAN=""
  RESET=""
fi

say() {
  printf "%s\n" "$*"
}

info() {
  say "${CYAN}$*${RESET}"
}

success() {
  say "${GREEN}$*${RESET}"
}

warn() {
  say "${YELLOW}$*${RESET}"
}

usage() {
  cat <<'EOF'
qualiax installer

Friendly setup script for local development and first-run demos.

Usage:
  ./install.sh [options]

Profiles:
  --minimal            Base package only.
  --recommended        Base package + broad audio format support.
  --all                Full stable install (default).

Extras:
  --audio              Add the `audio` extra.
  --perceptual         Add the `perceptual` extra.
  --live               Add the `live` extra.
  --watch              Add the `watch` extra.
  --ml                 Add the `ml` extra.

Optional add-ons:
  --with-torch         Also install `torch` and `torchaudio`.
  --with-crepe         Also install `crepe` manually for the neural F0 backend.
  --demo               Run `./demo.sh` after installation succeeds.

Environment:
  --global             Install into the current Python environment instead of a venv.
  --venv PATH          Virtualenv path to create/use (default: ./.venv).
  --python PATH        Python executable to use (default: python3).
  --help               Show this help.

Examples:
  ./install.sh
  ./install.sh --recommended
  ./install.sh --minimal --with-torch
  ./install.sh --audio --watch --demo
  ./install.sh --global --all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minimal)
      PROFILE="minimal"
      ;;
    --recommended)
      PROFILE="recommended"
      ;;
    --all)
      PROFILE="all"
      ;;
    --audio|--perceptual|--live|--watch|--ml)
      EXTRA_NAME="${1#--}"
      if [[ "$PROFILE" == "minimal" ]]; then
        PROFILE="$EXTRA_NAME"
      elif [[ "$PROFILE" == "recommended" ]]; then
        PROFILE="audio,$EXTRA_NAME"
      elif [[ "$PROFILE" == "all" ]]; then
        PROFILE="$EXTRA_NAME"
      else
        PROFILE="$PROFILE,$EXTRA_NAME"
      fi
      ;;
    --with-torch)
      INSTALL_TORCH=1
      ;;
    --with-crepe)
      INSTALL_CREPE=1
      ;;
    --demo)
      RUN_DEMO=1
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
      say "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  say "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

normalize_extras() {
  local raw="$1"
  local unique=()
  local item
  local seen
  if [[ -z "$raw" || "$raw" == "minimal" ]]; then
    printf '%s' ""
    return
  fi
  IFS=',' read -r -a items <<< "$raw"
  for item in "${items[@]}"; do
    [[ -z "$item" ]] && continue
    seen=0
    for existing in "${unique[@]:-}"; do
      if [[ "$existing" == "$item" ]]; then
        seen=1
        break
      fi
    done
    if [[ $seen -eq 0 ]]; then
      unique+=("$item")
    fi
  done
  (
    IFS=','
    printf '%s' "${unique[*]}"
  )
}

case "$PROFILE" in
  minimal)
    EXTRAS=""
    PROFILE_LABEL="minimal"
    ;;
  recommended)
    EXTRAS="audio"
    PROFILE_LABEL="recommended"
    ;;
  all)
    EXTRAS="all"
    PROFILE_LABEL="full"
    ;;
  *)
    EXTRAS="$(normalize_extras "$PROFILE")"
    PROFILE_LABEL="custom"
    ;;
esac

PACKAGE_SPEC="-e $SCRIPT_DIR"
if [[ -n "$EXTRAS" ]]; then
  PACKAGE_SPEC="-e $SCRIPT_DIR[$EXTRAS]"
fi

say "${BOLD}qualiax installer${RESET}"
say "${DIM}Perceptual speech and audio quality analyzer${RESET}"
say
say "Install profile: ${BOLD}$PROFILE_LABEL${RESET}${EXTRAS:+ (${EXTRAS})}"
say "Python: ${BOLD}$PYTHON_BIN${RESET}"

if command -v ffmpeg >/dev/null 2>&1; then
  say "ffmpeg: ${BOLD}available${RESET} ($(ffmpeg -version 2>&1 | head -1))"
else
  warn "ffmpeg not found."
  say "  MP3, AAC/M4A, and video-container decoding may be limited until you install it."
  say "  macOS:  brew install ffmpeg"
  say "  Debian: sudo apt install ffmpeg"
fi

if [[ $USE_GLOBAL -eq 1 ]]; then
  PYTHON_EXE="$PYTHON_BIN"
  ENV_LABEL="current Python environment"
else
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
    ENV_LABEL="active virtualenv at $VIRTUAL_ENV"
  else
    if [[ ! -d "$VENV_DIR" ]]; then
      info "Creating virtualenv at $VENV_DIR"
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    PYTHON_EXE="$VENV_DIR/bin/python"
    ENV_LABEL="virtualenv at $VENV_DIR"
  fi
fi

say "Install target: ${BOLD}$ENV_LABEL${RESET}"
say
info "Upgrading packaging tools..."
"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel

info "Installing qualiax..."
"$PYTHON_EXE" -m pip install $PACKAGE_SPEC

if [[ $INSTALL_TORCH -eq 1 ]]; then
  info "Installing PyTorch support..."
  "$PYTHON_EXE" -m pip install torch torchaudio
fi

if [[ $INSTALL_CREPE -eq 1 ]]; then
  info "Installing CREPE..."
  "$PYTHON_EXE" -m pip install crepe
fi

info "Running import verification..."
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
        module = importlib.import_module(mod)
        version = getattr(module, "__version__", "?")
        print(f"  ok  {label} {version}")
    except Exception as exc:
        failed.append((label, exc))

if failed:
    print("")
    for label, exc in failed:
        print(f"  fail {label}: {exc}")
    sys.exit(1)
PY

QUALIAX_CMD=""
if [[ $USE_GLOBAL -eq 1 ]]; then
  QUALIAX_CMD="qualiax"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  QUALIAX_CMD="$VIRTUAL_ENV/bin/qualiax"
else
  QUALIAX_CMD="$VENV_DIR/bin/qualiax"
fi

say
success "Installation complete."
say
say "${BOLD}Next steps${RESET}"
if [[ $USE_GLOBAL -eq 0 && -z "${VIRTUAL_ENV:-}" ]]; then
  say "1. Activate the environment:"
  say "   source \"$VENV_DIR/bin/activate\""
  say "2. Check the CLI:"
  say "   qualiax --help"
else
  say "1. Check the CLI:"
  say "   $QUALIAX_CMD --help"
fi
say "3. Try the sample file:"
say "   $QUALIAX_CMD sample.wav"
say "4. Run the demo script:"
say "   ./demo.sh"

if [[ $RUN_DEMO -eq 1 ]]; then
  say
  info "Running demo..."
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    QUALIAX_BIN="$VIRTUAL_ENV/bin/qualiax"
    QUALIAX_PYTHON="$VIRTUAL_ENV/bin/python"
  elif [[ $USE_GLOBAL -eq 0 ]]; then
    QUALIAX_BIN="$VENV_DIR/bin/qualiax"
    QUALIAX_PYTHON="$VENV_DIR/bin/python"
  else
    QUALIAX_BIN=""
    QUALIAX_PYTHON="$PYTHON_EXE"
  fi
  if [[ -n "$QUALIAX_BIN" ]]; then
    QUALIAX_BIN="$QUALIAX_BIN" QUALIAX_PYTHON="$QUALIAX_PYTHON" "$SCRIPT_DIR/demo.sh"
  else
    QUALIAX_PYTHON="$QUALIAX_PYTHON" "$SCRIPT_DIR/demo.sh"
  fi
fi
