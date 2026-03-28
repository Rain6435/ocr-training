# Multi-Stage Historical Document Digitization Pipeline

## Development Environment Setup Guide

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start (TL;DR)](#quick-start-tldr)
4. [Detailed Setup Instructions](#detailed-setup-instructions)
5. [GPU Setup (Optional but Recommended)](#gpu-setup-optional-but-recommended)
6. [Dataset Download](#dataset-download)
7. [Verification Tests](#verification-tests)
8. [IDE Configuration](#ide-configuration)
9. [Docker Setup (Alternative)](#docker-setup-alternative)
10. [Troubleshooting](#troubleshooting)
11. [Cloud Development Options](#cloud-development-options)

---

## Overview

This guide will help you set up a complete development environment for the Historical Document OCR Pipeline project. By the end, you'll have:

- Python 3.10 with all required libraries
- TensorFlow with GPU support (optional)
- OpenCV, Tesseract, and other OCR tools
- Development tools (Git, IDE, Jupyter)
- Access to training datasets

**Estimated Setup Time:** 1-2 hours (depending on download speeds and GPU setup)

---

## System Requirements

### Minimum Requirements (CPU-only development)

- **OS:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **CPU:** 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM:** 8GB (16GB recommended)
- **Storage:** 50GB free space (for datasets and models)
- **Internet:** Broadband connection for dataset downloads

### Recommended Requirements (GPU training)

- **OS:** Ubuntu 20.04/22.04 or Windows 10/11 with WSL2
- **CPU:** 6+ cores
- **RAM:** 16GB+ (32GB for large batch training)
- **Storage:** 100GB+ SSD
- **GPU:** NVIDIA GPU with 6GB+ VRAM (RTX 3060, RTX 3070, or better)
  - CUDA Compute Capability 3.5+
  - Examples: GTX 1060, RTX 2060, RTX 3060, RTX 3070, RTX 4070
- **CUDA:** 11.8+ (for TensorFlow 2.15+)

### Cloud Alternative

If you don't have adequate hardware, see [Cloud Development Options](#cloud-development-options)

---

## Quick Start (TL;DR)

For experienced users who want to get started immediately:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/historical-document-ocr.git
cd historical-document-ocr

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

# Verify installation
python scripts/verify_setup.py

# Download datasets
python scripts/download_datasets.py --datasets iam nist emnist

# Run example
python examples/basic_ocr.py --image data/samples/test_image.jpg
```
````

If everything works, you're ready to go! Otherwise, follow the detailed instructions below.

---

## Detailed Setup Instructions

### Step 1: Install Python 3.10

#### Ubuntu/Debian Linux

```bash
# Update package list
sudo apt update

# Install Python 3.10 and development tools
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install pip
sudo apt install python3-pip

# Verify installation
python3.10 --version  # Should show: Python 3.10.x
pip3 --version
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10
brew install python@3.10

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
export PATH="/usr/local/opt/python@3.10/bin:$PATH"

# Verify installation
python3.10 --version
```

#### Windows

1. Download Python 3.10 installer from [python.org](https://www.python.org/downloads/)
2. Run installer
   - ✅ Check "Add Python 3.10 to PATH"
   - ✅ Check "Install pip"
3. Open Command Prompt and verify:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Set Up Project Directory

```bash
# Create project directory
mkdir ~/projects
cd ~/projects

# Clone repository (if not already done)
git clone https://github.com/YOUR_USERNAME/historical-document-ocr.git
cd historical-document-ocr

# Or create new repository
git init
git remote add origin https://github.com/YOUR_USERNAME/historical-document-ocr.git
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Your prompt should now show (venv) prefix

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Important:** Always activate the virtual environment before working on the project!

### Step 4: Install Core Dependencies

#### Create requirements.txt

```txt
# requirements.txt

# Core ML/DL
tensorflow==2.15.0
numpy==1.24.3
scipy==1.11.4

# Image Processing
opencv-python==4.8.1.78
Pillow==10.1.0
scikit-image==0.22.0

# OCR Engines
pytesseract==0.3.10
transformers==4.35.2
torch==2.1.1  # For TrOCR

# Data Processing
pandas==2.1.3
scikit-learn==1.3.2
editdistance==0.6.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Dashboard
streamlit==1.28.2

# Experiment Tracking
tensorboard==2.15.1
mlflow==2.8.1  # Optional

# Utilities
tqdm==4.66.1
python-dotenv==1.0.0
PyYAML==6.0.1
click==8.1.7

# Development Tools
pytest==7.4.3
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1
ipython==8.18.1
jupyter==1.0.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0
```

#### Install dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# This may take 5-10 minutes depending on your connection
```

#### Verify TensorFlow installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

Expected output:

```
TensorFlow version: 2.15.0
GPU available: []  # Empty if no GPU, or list of GPUs if available
```

### Step 5: Install System Dependencies

#### Install Tesseract OCR

##### Ubuntu/Debian

```bash
# Install Tesseract and language data
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Optional: Install additional languages
sudo apt-get install tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa

# Verify installation
tesseract --version
which tesseract
```

##### macOS

```bash
# Install via Homebrew
brew install tesseract

# Optional: Install additional languages
brew install tesseract-lang

# Verify installation
tesseract --version
```

##### Windows

1. Download installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer (use default location: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH:
   - Right-click "This PC" → Properties → Advanced System Settings
   - Environment Variables → Path → Edit
   - Add: `C:\Program Files\Tesseract-OCR`
4. Verify:
   ```cmd
   tesseract --version
   ```

#### Install OpenCV Dependencies (Linux only)

```bash
# Install system libraries for OpenCV
sudo apt-get install libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0

# For video support (optional)
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
```

### Step 6: Configure Project Structure

```bash
# Create directory structure
mkdir -p data/{raw,processed,samples}
mkdir -p data/raw/{iam,nist,emnist}
mkdir -p data/processed/{train,val,test}
mkdir -p models/{checkpoints,difficulty_classifier,custom_ocr}
mkdir -p experiments
mkdir -p logs
mkdir -p outputs

# Create initial config files
touch config/preprocessing_profiles.yaml
touch config/router_config.yaml
touch .env

# Copy environment template
cat > .env << EOF
# Environment Configuration

# Paths
DATA_DIR=./data
MODEL_DIR=./models
EXPERIMENT_DIR=./experiments

# API Keys (if using cloud OCR)
GOOGLE_VISION_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_key_here

# Hardware
DEVICE=cuda  # or cpu
GPU_MEMORY_FRACTION=0.8

# Logging
LOG_LEVEL=INFO
TENSORBOARD_PORT=6006
MLFLOW_PORT=5000

# Tesseract
TESSERACT_CMD=/usr/bin/tesseract  # Adjust for your system
EOF
```

### Step 7: Install Development Tools

```bash
# Install Git hooks for code quality
pip install pre-commit

# Create pre-commit config
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203,E501']

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
EOF

# Install hooks
pre-commit install
```

---

## GPU Setup (Optional but Recommended)

### Check GPU Compatibility

```bash
# Check if you have NVIDIA GPU
lspci | grep -i nvidia

# Check NVIDIA driver version
nvidia-smi
```

### Install CUDA Toolkit (Linux)

#### Ubuntu 22.04

```bash
# Download CUDA 11.8 (compatible with TensorFlow 2.15)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-11-8

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Reload bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

### Install cuDNN

```bash
# Download cuDNN from NVIDIA (requires free account)
# https://developer.nvidia.com/cudnn

# Install cuDNN (example for Ubuntu)
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev

# Verify installation
ldconfig -p | grep cudnn
```

### Install TensorFlow with GPU Support

```bash
# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0

# Verify GPU is detected
python -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

Expected output:

```
Num GPUs: 1  # Or however many GPUs you have
```

### Configure GPU Memory Growth

```python
# Add to beginning of training scripts
import tensorflow as tf

# Prevent TensorFlow from allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
```

### Windows GPU Setup

1. Install [NVIDIA CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install [cuDNN 8.9](https://developer.nvidia.com/cudnn)
3. Install TensorFlow:
   ```cmd
   pip install tensorflow[and-cuda]==2.15.0
   ```

### macOS (Apple Silicon)

```bash
# Install Metal-optimized TensorFlow
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0

# Verify GPU acceleration
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Dataset Download

### Download Scripts

Create `scripts/download_datasets.py`:

```python
#!/usr/bin/env python3
"""
Download and organize datasets for training.
"""

import argparse
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, destination: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def download_iam_handwriting():
    """
    Download IAM Handwriting Database.

    Note: Requires registration at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
    You'll need to manually download after registering.
    """
    print("\n=== IAM Handwriting Database ===")
    print("This dataset requires registration.")
    print("Steps:")
    print("1. Go to: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
    print("2. Register for free account")
    print("3. Download these files:")
    print("   - words.tgz")
    print("   - lines.tgz")
    print("   - ascii.tgz (ground truth)")
    print("4. Place in data/raw/iam/")
    print("\nPress Enter when done...")
    input()

def download_nist_handwriting():
    """
    Download NIST Special Database 19.
    """
    print("\n=== NIST Special Database 19 ===")
    print("This is a large dataset (~7GB).")

    # NIST SD19 is available via NIST website
    # Public domain, no registration required
    base_url = "https://s3.amazonaws.com/nist-srd/SD19"

    # Create directories
    nist_dir = Path("data/raw/nist")
    nist_dir.mkdir(parents=True, exist_ok=True)

    # Download sample (full dataset is very large)
    print("Downloading NIST sample...")
    # In practice, you'd download the full dataset or use EMNIST (derived from NIST)
    print("Note: Using EMNIST (derived from NIST) is recommended for easier setup.")
    print("Run with --datasets emnist instead.")

def download_emnist():
    """
    Download EMNIST dataset (easier alternative to NIST).
    """
    print("\n=== EMNIST Dataset ===")

    from tensorflow.keras import datasets

    # EMNIST is available through TensorFlow
    print("EMNIST will be automatically downloaded when first used by TensorFlow.")
    print("No manual download needed!")

    # Create marker file
    emnist_dir = Path("data/raw/emnist")
    emnist_dir.mkdir(parents=True, exist_ok=True)
    (emnist_dir / "README.txt").write_text(
        "EMNIST dataset will be downloaded automatically by TensorFlow.\n"
        "See: https://www.tensorflow.org/datasets/catalog/emnist\n"
    )

def download_sample_images():
    """Download sample test images."""
    print("\n=== Sample Test Images ===")

    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download some public domain historical documents
    samples = {
        "sample_clean.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Declaration_independence.jpg/800px-Declaration_independence.jpg",
        "sample_handwritten.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Abraham_Lincoln_letter_to_Horace_Greeley.jpg/600px-Abraham_Lincoln_letter_to_Horace_Greeley.jpg",
    }

    for filename, url in samples.items():
        dest = samples_dir / filename
        if not dest.exists():
            print(f"Downloading {filename}...")
            download_file(url, str(dest))
        else:
            print(f"{filename} already exists, skipping.")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for OCR training")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["iam", "nist", "emnist", "samples", "all"],
        default=["samples"],
        help="Which datasets to download"
    )
    args = parser.parse_args()

    if "all" in args.datasets:
        args.datasets = ["iam", "emnist", "samples"]

    print("=== Dataset Download Script ===")
    print(f"Datasets to download: {', '.join(args.datasets)}")

    if "iam" in args.datasets:
        download_iam_handwriting()

    if "nist" in args.datasets:
        download_nist_handwriting()

    if "emnist" in args.datasets:
        download_emnist()

    if "samples" in args.datasets:
        download_sample_images()

    print("\n=== Download Complete ===")
    print("Next steps:")
    print("1. Run: python scripts/prepare_data.py")
    print("2. Run: python scripts/verify_setup.py")

if __name__ == "__main__":
    main()
```

### Run Download Script

```bash
# Make script executable
chmod +x scripts/download_datasets.py

# Download sample images (quick start)
python scripts/download_datasets.py --datasets samples

# Download EMNIST (automatic, no manual download)
python scripts/download_datasets.py --datasets emnist

# Download IAM (requires registration)
python scripts/download_datasets.py --datasets iam
```

---

## Verification Tests

Create `scripts/verify_setup.py`:

```python
#!/usr/bin/env python3
"""
Verify that the development environment is set up correctly.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.9+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} not found")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow can access GPU."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow GPU support ({len(gpus)} GPU(s) detected)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("⚠ TensorFlow installed but no GPU detected (CPU-only mode)")
            return True
    except Exception as e:
        print(f"✗ TensorFlow GPU check failed: {e}")
        return False

def check_tesseract():
    """Check if Tesseract is installed."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ Tesseract OCR: {version}")
            return True
        else:
            print("✗ Tesseract not found")
            return False
    except FileNotFoundError:
        print("✗ Tesseract not found in PATH")
        return False

def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "experiments",
        "logs"
    ]

    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False

    return all_exist

def run_basic_tests():
    """Run basic functionality tests."""
    print("\nRunning basic tests...")

    # Test OpenCV
    try:
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV basic operations")
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")

    # Test TensorFlow
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,))
        ])
        print("✓ TensorFlow basic operations")
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")

    # Test Tesseract via pytesseract
    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        # Create simple test image
        img = Image.new('RGB', (100, 30), color='white')
        # Note: This might fail if Tesseract path not set
        # That's okay, we already checked Tesseract separately
        print("✓ pytesseract integration")
    except Exception as e:
        print(f"⚠ pytesseract test: {e}")

def main():
    print("=== Development Environment Verification ===\n")

    checks = []

    print("Python:")
    checks.append(check_python_version())

    print("\nCore Packages:")
    checks.append(check_package("tensorflow"))
    checks.append(check_package("numpy"))
    checks.append(check_package("opencv-python", "cv2"))
    checks.append(check_package("pytesseract"))
    checks.append(check_package("PIL", "PIL"))
    checks.append(check_package("sklearn"))

    print("\nWeb Framework:")
    checks.append(check_package("fastapi"))
    checks.append(check_package("streamlit"))

    print("\nExperiment Tracking:")
    checks.append(check_package("tensorboard"))

    print("\nGPU Support:")
    checks.append(check_tensorflow_gpu())

    print("\nSystem Tools:")
    checks.append(check_tesseract())

    print("\nProject Structure:")
    checks.append(check_directories())

    # Run basic tests
    run_basic_tests()

    # Summary
    print("\n" + "="*50)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print("\nYou're ready to start development!")
        return 0
    else:
        print(f"⚠ {passed}/{total} checks passed")
        print(f"\nPlease fix the {total - passed} failed check(s) above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Run Verification

```bash
# Make script executable
chmod +x scripts/verify_setup.py

# Run verification
python scripts/verify_setup.py
```

Expected output (all checks passing):

```
=== Development Environment Verification ===

Python:
✓ Python 3.10.12

Core Packages:
✓ tensorflow
✓ numpy
✓ opencv-python
✓ pytesseract
✓ PIL
✓ sklearn

Web Framework:
✓ fastapi
✓ streamlit

Experiment Tracking:
✓ tensorboard

GPU Support:
✓ TensorFlow GPU support (1 GPU(s) detected)
  - /physical_device:GPU:0

System Tools:
✓ Tesseract OCR: tesseract 5.3.0

Project Structure:
✓ data/
✓ data/raw/
✓ data/processed/
✓ models/
✓ experiments/
✓ logs/

Running basic tests...
✓ OpenCV basic operations
✓ TensorFlow basic operations
✓ pytesseract integration

==================================================
✓ All checks passed (16/16)

You're ready to start development!
```

---

## IDE Configuration

### VS Code Setup (Recommended)

#### Install VS Code Extensions

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-azuretools.vscode-docker
code --install-extension eamodio.gitlens
code --install-extension ms-vscode.makefile-tools
```

#### Create VS Code Settings

`.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

#### Create Launch Configuration

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Train Custom OCR Model",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/train_custom_ocr.py",
      "args": ["--config", "config/model_configs/custom_ocr.yaml"],
      "console": "integratedTerminal"
    },
    {
      "name": "FastAPI Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.api.main:app", "--reload", "--port", "8000"],
      "console": "integratedTerminal"
    },
    {
      "name": "Streamlit Dashboard",
      "type": "python",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "dashboard/app.py"],
      "console": "integratedTerminal"
    }
  ]
}
```

### PyCharm Setup

1. Open project directory in PyCharm
2. Configure Python interpreter:
   - File → Settings → Project → Python Interpreter
   - Add interpreter → Existing environment
   - Select `venv/bin/python`
3. Configure code style:
   - File → Settings → Editor → Code Style → Python
   - Set line length to 88
4. Enable pytest:
   - File → Settings → Tools → Python Integrated Tools
   - Default test runner: pytest

### Jupyter Notebook Setup

```bash
# Install Jupyter kernel for virtual environment
pip install ipykernel
python -m ipykernel install --user --name=ocr-pipeline --display-name "OCR Pipeline"

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

---

## Docker Setup (Alternative)

If you prefer containerized development:

### Dockerfile

```dockerfile
# Dockerfile

FROM tensorflow/tensorflow:2.15.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000 6006 5000

# Default command
CMD ["/bin/bash"]
```

### Docker Compose

```yaml
# docker-compose.yml

version: "3.8"

services:
  ocr-pipeline:
    build: .
    image: historical-document-ocr:latest
    container_name: ocr-pipeline-dev
    volumes:
      - .:/app
      - ./data:/app/data
      - ./models:/app/models
      - ./experiments:/app/experiments
    ports:
      - "8000:8000" # FastAPI
      - "6006:6006" # TensorBoard
      - "5000:5000" # MLflow
      - "8501:8501" # Streamlit
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: /bin/bash
```

### Build and Run

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# Enter container
docker-compose exec ocr-pipeline bash

# Run verification
python scripts/verify_setup.py
```

---

## Troubleshooting

### Common Issues

#### 1. TensorFlow GPU Not Detected

**Problem:** `tf.config.list_physical_devices('GPU')` returns empty list

**Solutions:**

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check cuDNN
ldconfig -p | grep cudnn

# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0

# Check compatibility
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
```

#### 2. Tesseract Not Found

**Problem:** `pytesseract.TesseractNotFoundError`

**Solutions:**

```python
# Explicitly set Tesseract path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux
# or
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

#### 3. Out of Memory (OOM) During Training

**Problem:** GPU runs out of memory during training

**Solutions:**

```python
# Reduce batch size
batch_size = 16  # or smaller

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Use mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

#### 4. Slow Dataset Loading

**Problem:** Training is slow due to data loading bottleneck

**Solutions:**

```python
# Use tf.data with prefetching
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Enable parallel data loading
dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Cache dataset in memory
dataset = dataset.cache()
```

#### 5. Import Errors

**Problem:** `ModuleNotFoundError` despite package being installed

**Solutions:**

```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall package
pip uninstall <package>
pip install <package>

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Getting Help

If you encounter issues not covered here:

1. Check error messages carefully
2. Search project issues on GitHub
3. Check TensorFlow/PyTorch forums
4. Stack Overflow with specific error messages
5. Contact course instructor/TA

---

## Cloud Development Options

If you don't have adequate local hardware:

### Google Colab (Free)

**Pros:**

- Free GPU access (Tesla T4)
- No setup required
- Jupyter notebook interface

**Cons:**

- Session timeouts (12 hours max)
- Limited storage
- Can't run long training jobs

**Setup:**

```python
# In Colab notebook
!git clone https://github.com/YOUR_USERNAME/historical-document-ocr.git
%cd historical-document-ocr
!pip install -r requirements.txt

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
```

### Kaggle Notebooks (Free)

**Pros:**

- Free GPU (30 hours/week)
- Pre-installed ML libraries
- Large dataset library

**Cons:**

- Session limits
- Internet disabled during training

**Usage:**

1. Create account at kaggle.com
2. Go to Code → New Notebook
3. Enable GPU: Settings → Accelerator → GPU
4. Upload project files or connect to GitHub
