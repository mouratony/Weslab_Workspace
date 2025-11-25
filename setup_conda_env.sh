#!/usr/bin/env bash
set -e

ENV_NAME="wesl_tony_env"
PY_VERSION="3.11"
REQ_FILE="requirements.txt"

# Detect OS and architecture
OS_NAME="$(uname -s 2>/dev/null || echo "unknown")"
ARCH_NAME="$(uname -m 2>/dev/null || echo "unknown")"

echo "Detected OS: ${OS_NAME}"
echo "Detected architecture: ${ARCH_NAME}"

# Basic hint for Windows users running outside WSL/Git Bash
case "$OS_NAME" in
  MINGW*|MSYS*|CYGWIN*)
    echo "Note: You appear to be on Windows (MSYS/MINGW)."
    echo "This script works best in WSL or Git Bash with conda installed."
    ;;
esac

# Check conda
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda or Anaconda and ensure 'conda' is on your PATH."
  exit 1
fi

# Enable 'conda activate' in this shell
eval "$(conda shell.bash hook)"

# Create env if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists."
else
  echo "Creating conda env '$ENV_NAME' with Python $PY_VERSION ..."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

echo "Activating env '$ENV_NAME' ..."
conda activate "$ENV_NAME"

# Install dependencies
if [ -f "$REQ_FILE" ]; then
  echo "Installing dependencies from $REQ_FILE ..."
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ_FILE"
else
  echo "Warning: $REQ_FILE not found. Skipping pip install."
fi

echo
echo "Done. Environment Created Successfully!"
echo "To use this environment later, run:"
echo "  conda activate $ENV_NAME"
