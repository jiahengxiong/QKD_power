#!/bin/bash

# 尝试激活 Poetry 环境
if command -v poetry &> /dev/null; then
    echo "Using Poetry environment..."
    poetry run python train_cma.py
    exit 0
fi

# 尝试激活 Conda 环境 (假设环境名为 qkd 或者 base)
# 你可以修改下面的环境名
CONDA_ENV_NAME="qkd"
if command -v conda &> /dev/null; then
    echo "Attempting to activate Conda environment: $CONDA_ENV_NAME"
    # Conda init hack for shell script
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    if python -c "import torch" &> /dev/null; then
        echo "Successfully activated Conda environment with torch."
        python train_cma.py
        exit 0
    else
        echo "Conda environment '$CONDA_ENV_NAME' does not have torch or does not exist."
        # Fallback to base
        conda activate base
    fi
fi

# 尝试直接运行 (如果用户已经在正确的环境里)
echo "Trying default python..."
if python -c "import torch" &> /dev/null; then
    python train_cma.py
    exit 0
elif python3 -c "import torch" &> /dev/null; then
    python3 train_cma.py
    exit 0
else
    echo "❌ Error: Could not find a python environment with 'torch' installed."
    echo "Please activate your virtual environment manually."
    exit 1
fi
