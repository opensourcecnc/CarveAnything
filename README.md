# CarveAnything

A Python pipeline designed to transform 2D bas relief images into detailed 3D bas-reliefs using Depth Anything 3 depth estimation.

## Features

* **Depth Estimation:** Uses Depth Anything 3 for highly detailed depth map prediction.
* **Automated 3D Relief Generation:** Converts depth map directly into a mesh (OBJ).
* **Interactive CLI Interface:** Simple, prompt-guided execution with command-line arguments.
* **GPU Accelerated:** Optimized for CUDA-enabled PyTorch execution.
* **CNC / 3D-Printing Ready:** Designed to generate usable 3D geometries for CAM software, carving, and 3D printing.

## Prerequisites

Before setting up `CarveAnything`, ensure you have the following installed on your host system:

* **Python:** 3.10 or higher recommended (Python 3.9 minimum)
* **Git:** Installed and configured
* **NVIDIA GPU Driver:** Version `550.58` or higher (for CUDA 12.4 support)

## Installation

Follow these step-by-step instructions to set up `CarveAnything` on your local system:

### 1. Clone the Repository

Open your terminal or PowerShell and clone the repository:

```bash
git clone https://github.com/opensourcecnc/CarveAnything.git
cd CarveAnything
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Upgrade `pip` and install all necessary Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch / CUDA:** `requirements.txt` configures **CUDA 12.4** PyTorch builds by default. If running on CPU only or requiring a different CUDA release, refer to the [PyTorch Get Started guide](https://pytorch.org/get-started/locally/).

## Model Weights Setup

`CarveAnything` requires model weights from the Hugging Face `DA3MONO-LARGE` repository to perform depth estimation.

1.The Hugging Face repository: [depth-anything/DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE/tree/main)

2. Download the following files:
   * `config.json`
   * `model.safetensors`

3. Place the downloaded files into the `da-models/` directory in the root of the repository.

## Usage

`CarveAnything` features an interactive terminal workflow. You do not need to pass command-line arguments when launching the program.

Simply execute the main script:

```bash
python CarveAnything.py
```

Once launched, the script will guide you with step-by-step on-screen prompts to input:

Generated output files (depth maps and 3D mesh files) will be saved automatically in the `output/` directory.

## License

Distributed under the MIT License. See `LICENSE` for details.