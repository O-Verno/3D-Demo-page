#!/bin/bash  # Declare the script interpreter

# Detect operating system type (Linux, macOS, Windows with Git Bash)
OS_TYPE=$(uname -s)

# Choose Python and activation folder based on OS
if [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "Darwin" ]]; then
    PYTHON=python3         # Use python3 for Unix-based systems
    ACTIVATE=bin           # venv activation path
else
    PYTHON=python          # Use python for Windows
    ACTIVATE=Scripts       # venv activation path on Windows
fi

# Check if virtual environment already exists
if [ ! -d ".venv" ]; then
    # Create virtual environment
    $PYTHON -m venv .venv

    # Upgrade pip inside the virtual environment
    .venv/"$ACTIVATE"/"$PYTHON" -m pip install --upgrade pip

    # Install main project dependencies
    .venv/"$ACTIVATE"/pip install -r requirements.txt

    # Download Deepskin source code from GitHub
    curl -L -o Deepskin.zip https://github.com/Nico-Curti/Deepskin/archive/refs/heads/main.zip

    # Unzip downloaded file
    unzip Deepskin.zip

    # Rename folder for consistency
    mv Deepskin-main Deepskin

    # Remove downloaded zip file
    rm -f Deepskin.zip

    # Install Deepskin dependencies
    .venv/"$ACTIVATE"/python -m pip install -r ./Deepskin/requirements.txt

    # Install Deepskin package itself
    .venv/"$ACTIVATE"/python -m pip install ./Deepskin

    # Remove Deepskin source after install to keep project clean
    rm -rf ./Deepskin
else
    # Message shown if virtual environment already exists
    echo "There is already a .venv, if you want to re-init, delete it first (rm -rf .venv)"
fi
