# Image Deblurring Using Regularization and Spectral Methods

**Author:** Serge Francis Ineza N.  
**Course:** Scientific Computing II

## Installation
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
scipy>=1.7.0
scikit-image>=0.18.0
matplotlib>=3.4.0
Pillow>=8.3.0
opencv-python>=4.5.0
pandas>=1.3.0
scikit-learn>=0.24.0
```

## Setup

1. Download test images (automatic):
```bash
cd code
python download_test_images.py
```

2. Run the main experiment:
```bash
python main_experiment.py
```

## What to Expect

The program will:
- Generate blurred versions of your test images
- Apply Tikhonov regularization and TSVD deblurring methods
- Compare both methods and save results

## Results Location

- **Figures and plots:** `results/figures/`
- **Reconstructed images:** `data/deblurred/`
- **Metrics (CSV):** `results/experiment_results.csv`

## Implementation Detail: Image Downsampling

To keep the explicit convolution matrices and SVD computations feasible on a laptop, all images are automatically downsampled to a maximum size of **64×64** pixels when they are loaded.

This is implemented in `code/blur.generation.py` in the `load_image_grayscale` function. Any image whose larger side exceeds 64 pixels is resized so that its longest side is 64, using bicubic interpolation.

The motivation is that a full 2D convolution matrix for a 512×512 image would have size (512²)×(512²), requiring hundreds of gigabytes of RAM and making direct Tikhonov and SVD-based methods impractical. Using 64×64 images keeps the matrices around 4096×4096, which fits in memory and allows us to run L-curve and TSVD experiments.

If you want to experiment with higher resolutions, you can increase the `max_side` parameter in `load_image_grayscale`, but memory usage and runtime will grow quickly.

## Quick Test (Individual Methods)

**Tikhonov only:**
```bash
cd code
python -c "from main_experiment import test_tikhonov_method; test_tikhonov_method()"
```

**TSVD only:**
```bash
cd code
python -c "from main_experiment import test_tsvd_method; test_tsvd_method()"
```

**Image Downsampling Note**

Images are downsampled to 64x64 in `load_image_grayscale` for computational reasons. To change this, modify the `max_side` parameter in `load_image_grayscale` in `code/blur.generation.py`. Note that increasing this value will significantly impact memory usage and runtime.