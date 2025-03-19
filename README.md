
# Installation Instructions for Traffic Sign Classification Project

This document provides step-by-step instructions to set up the required Python packages and dependencies for running and training the Traffic Sign Classification project utilizing PyTorch, OpenCV, Pillow, SciPy, and NumPy.

## Prerequisites
- Python (Recommended: 3.8 or later)
- pip (Python Package Installer)

---

## Installing PyTorch

PyTorch can be installed with either CPU-only support or GPU support. For basic functionality and assignments, the CPU-only version is sufficient. However, if your system is equipped with an NVIDIA GPU, it is recommended to install the GPU-supported version of PyTorch for enhanced performance.

### CPU-only Version

To install the CPU-only version of PyTorch, follow these steps:

1. Open the official PyTorch installation page: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. Select the following options:
    - PyTorch Build: Stable
    - Your Operating System (e.g., Linux, Windows, or macOS)
    - Package: pip
    - Language: Python
    - Compute Platform: CPU
3. Copy the provided installation command and execute it in your terminal. Typically, it looks like this:
```bash
pip install torch torchvision torchaudio
```

### GPU Version (for NVIDIA GPUs)

To utilize the GPU-enabled version of PyTorch:

1. Verify your NVIDIA GPU driver and CUDA version.
2. Visit the PyTorch installation page: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
3. Select the following:
    - PyTorch Build: Stable
    - Your Operating System (e.g., Linux, Windows)
    - Package: pip
    - Language: Python
    - Compute Platform: Select your CUDA version (e.g., CUDA 12.1)
4. Copy and run the provided pip installation command. It should look similar to:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
5. If any failure arises, please contact the tutor during lab.
### Verifying PyTorch Installation

Verify your PyTorch installation and GPU availability (if applicable) by running the following command:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If your GPU installation was successful, this command will return:
```
CUDA available: True
```

Otherwise, it will return:
```
CUDA available: False
```

---

## Installing Additional Python Packages

Install additional required Python packages using `pip` as follows:

### OpenCV
```bash
pip install opencv-python
```

### Pillow (PIL)
```bash
pip install pillow
```

### SciPy
```bash
pip install scipy
```

### NumPy
```bash
pip install numpy
```

---

## Quick Summary

Execute the commands below for a quick installation summary (assuming CPU-only installation):

```bash
pip install torch torchvision torchaudio
pip install opencv-python pillow scipy numpy
```

---

## Running the Code

After completing the installation, you can now proceed to run your training script:

```bash
python train_final.py
```

- Adjust the `--data_path` parameter according to the location of your dataset.
- Modify `--epochs` and `--batch_size` based on your preferences and hardware capabilities.

---

## Troubleshooting

- Ensure your Python and pip installations are correctly configured and up-to-date.
- If encountering CUDA-related errors, verify your NVIDIA driver and CUDA installation versions.
- For further assistance or installation troubleshooting, refer to the official documentation:
    - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
    - [OpenCV Documentation](https://docs.opencv.org/)
    - [Pillow Documentation](https://pillow.readthedocs.io/)
    - [SciPy Documentation](https://scipy.org/)
    - [NumPy Documentation](https://numpy.org/)
 
# Dataset Download
- [Google Drive]([https://pytorch.org/docs/stable/index.html](https://drive.google.com/drive/folders/1ZlGBDe9RKQqffznb6k4C6Px3ws5Sr_5A?usp=sharing))

---

Following these instructions ensures a stable and reproducible development environment for the Traffic Sign Classification project.
