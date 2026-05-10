#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_FILE="${1:-$SCRIPT_DIR/sample.wav}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/demo_out}"
QUALIAX_BIN="${QUALIAX_BIN:-}"
QUALIAX_PYTHON="${QUALIAX_PYTHON:-python3}"

if [[ ! -f "$SAMPLE_FILE" ]]; then
  echo "Sample file not found: $SAMPLE_FILE" >&2
  echo "Pass a WAV path explicitly, for example:" >&2
  echo "  ./demo.sh /path/to/audio.wav" >&2
  exit 1
fi

resolve_qualiax_cmd() {
  if [[ -n "$QUALIAX_BIN" && -x "$QUALIAX_BIN" ]]; then
    printf '%s\n' "$QUALIAX_BIN"
    return
  fi
  if [[ -x "$SCRIPT_DIR/.venv/bin/qualiax" ]]; then
    printf '%s\n' "$SCRIPT_DIR/.venv/bin/qualiax"
    return
  fi
  if command -v qualiax >/dev/null 2>&1; then
    printf '%s\n' "qualiax"
    return
  fi
  printf '%s\n' "$QUALIAX_PYTHON"
}

if [[ -n "$QUALIAX_BIN" && -x "$QUALIAX_BIN" ]]; then
  QUALIAX_CMD=("$QUALIAX_BIN")
elif [[ -x "$SCRIPT_DIR/.venv/bin/qualiax" ]]; then
  QUALIAX_CMD=("$SCRIPT_DIR/.venv/bin/qualiax")
elif command -v qualiax >/dev/null 2>&1; then
  QUALIAX_CMD=("qualiax")
else
  QUALIAX_CMD=("$QUALIAX_PYTHON" "-m" "qualiax.cli")
fi

mkdir -p "$OUTPUT_DIR"
JSON_OUT="$OUTPUT_DIR/report.json"
MD_OUT="$OUTPUT_DIR/report.md"
HTML_OUT="$OUTPUT_DIR/report.html"

echo "qualiax demo"
echo "  input:  $SAMPLE_FILE"
echo "  output: $OUTPUT_DIR"
echo

echo "1. Console summary"
"${QUALIAX_CMD[@]}" "$SAMPLE_FILE" --metrics basic,loudness,speech
echo

echo "2. JSON report"
"${QUALIAX_CMD[@]}" "$SAMPLE_FILE" --output "$JSON_OUT" --silent
echo "   wrote: $JSON_OUT"
echo

echo "3. Markdown report"
"${QUALIAX_CMD[@]}" "$SAMPLE_FILE" --output "$MD_OUT" --silent
echo "   wrote: $MD_OUT"
echo

echo "4. HTML report"
"${QUALIAX_CMD[@]}" "$SAMPLE_FILE" --output "$HTML_OUT" --silent
echo "   wrote: $HTML_OUT"
echo

echo "Done. Open the outputs in $OUTPUT_DIR to inspect the results."
