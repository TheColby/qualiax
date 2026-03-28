#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing qualiax..."
pip install -e "$SCRIPT_DIR/[all]"
echo "Done. Run: qualiax --help"
