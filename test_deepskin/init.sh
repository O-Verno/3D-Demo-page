#!/bin/bash

OS_TYPE=$(uname -s)

if [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "Darwin" ]]; then
    PYTHON=python3
    ACTIVATE=bin
else
    PYTHON=python
    ACTIVATE=Scripts
fi

if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv

    .venv/"$ACTIVATE"/"$PYTHON" -m pip install --upgrade pip
    .venv/"$ACTIVATE"/pip install -r requirements.txt

    curl -L -o Deepskin.zip https://github.com/Nico-Curti/Deepskin/archive/refs/heads/main.zip
    unzip Deepskin.zip
    mv Deepskin-main Deepskin
    rm -f Deepskin.zip
    .venv/"$ACTIVATE"/python -m pip install -r ./Deepskin/requirements.txt
    .venv/"$ACTIVATE"/python -m pip install ./Deepskin
    rm -rf ./Deepskin
else
    echo "There is already a .venv, if you want to re-init, delete it first (rm -rf .venv)"
fi
