# GTU Digital Image Processing Project

White Blood Cell Detection and Analysis using Manual Image Processing Algorithms

## Project Overview

This project implements a complete image processing pipeline for detecting and analyzing white blood cells (WBCs) in microscope images using classical, manually implemented image processing algorithms.

## Features

- **Z-Stack Focus Analysis**: Selects the best-focused image from a Z-stack using classical focus measures
- **White Blood Cell Detection**: Detects and segments WBCs based on blue channel dominance
- **Cell Size Analysis**: Separates detected cells into large and small categories


## File Structure

```
├── main.py                 # Non-vectorized version
├── main_vectorized.py      # Optimized vectorized version
├── samples/                # Input images
└── results-main/           # Output directory
```

## Requirements

```bash
pip install numpy opencv-python matplotlib
```

## Usage

### Basic Usage
```bash
python main.py
```

### Quick Start
```python
from main import WBCellAnalysis, PipelineMode

wbca = WBCellAnalysis("path/to/image.jpg", scale=0.5)
wbca.analyze_image(
    scale=0.5,
    pipeline_mode=PipelineMode.SCALE_BLUR_BLUE,
    w_r=1.0,
    w_g=0.0
)
```

## Processing Pipelines

### Pipeline 1: SCALE_BLUR_BLUE
```
Original Image → Scale → Gaussian Blur → Blue Extraction → Segmentation
```

Performs scaling first, reducing computational cost for subsequent operations.

### Pipeline 2: BLUR_BLUE_SCALE
```
Original Image → Gaussian Blur → Blue Extraction → Scale → Segmentation
```

Performs blur and extraction on full-resolution image before scaling.

## Manual Algorithm Implementations

All algorithms are implemented using classical image processing techniques:

- **Image Scaling**: Nearest neighbor interpolation
- **Gaussian Blur**: 5×5 separable convolution kernel
- **Blue Channel Extraction**: `BlueDominance = B - w_r·R - w_g·G`
- **Otsu Thresholding**: Histogram-based automatic threshold calculation
- **Morphological Operations**: Erosion, Dilation, Opening, Closing
- **Connected Components Labeling**: DFS-based region labeling (4-connectivity)
- **Focus Measures**: Laplacian Variance, Tennengrad Gradient, Brenner Focus

## Configuration

Parameters can be adjusted via the `Config` class:

```python
class Config:
    SAMPLE_DIR = "samples/1212"
    RESULTS_DIR = "results-main"
    MIN_SIZE_LARGE_WBC = 400
    MAX_SIZE_LARGE_WBC = 2500
    CROP_AREA_OUT_WIDTH = 80
    CROP_AREA_OUT_HEIGHT = 80
```

## Input/Output Structure

### Input Images (Z-Stack)
The pipeline processes Z-stack microscope images. Example input structure:

```
samples/1212/
├── Image__2025-12-30__16-14-35.jpg
├── Image__2025-12-30__16-14-49.jpg
├── Image__2025-12-30__16-14-53.jpg
├── ... (16 images total at different focal planes)
└── Image__2025-12-30__16-16-32.jpg
```

### Output Structure

```
results-main/samples/1212/
├── original_image.png                    # Best focused image
├── focus_analysis_results.json           # Focus metrics for all images
├── focus_measure_curves.png              # Visualization of focus metrics
├── hybrid_focus_curve.png                # Combined focus score plot
├── cell_detection_results.txt            # Detection summary
├── scale_blur_blue/
│   ├── wr_1.0_wg_0.0/
│   ├── wr_0.9_wg_0.1/
│   └── wr_0.8_wg_0.2/
│       ├── blue_dom.png
│       ├── otsu_thresholded_blue_channel.png
│       ├── after_opening.png
│       ├── after_closing.png
│       ├── overlayed_labels.png
│       ├── large_white_blood_cells.png
│       ├── small_white_blood_cells.png
│       ├── cropped_large_cells/
│       │   ├── large_cell_1.png
│       │   ├── large_cell_2.png
│       │   └── ... (9 cells)
│       └── cropped_small_cells/
│           ├── small_cell_1.png
│           ├── small_cell_2.png
│           └── ... (175 cells)
└── blur_blue_scale/
    └── (same structure as scale_blur_blue)
```

### Detection Results

The pipeline tests multiple parameter combinations and generates detailed results:

| Configuration | Pipeline | Large WBCs | Small WBCs |
|--------------|----------|------------|------------|
| w_r=1.0, w_g=0.0 | SCALE_BLUR_BLUE | 9 | 168 |
| w_r=1.0, w_g=0.0 | BLUR_BLUE_SCALE | 9 | 192 |
| w_r=0.9, w_g=0.1 | SCALE_BLUR_BLUE | 9 | 160 |
| w_r=0.9, w_g=0.1 | BLUR_BLUE_SCALE | 9 | 183 |
| w_r=0.8, w_g=0.2 | SCALE_BLUR_BLUE | 9 | 156 |
| w_r=0.8, w_g=0.2 | BLUR_BLUE_SCALE | 9 | 175 |

**Note**: Large WBCs have area 200-2000 pixels, small WBCs have area <200 pixels. Each detected cell is automatically cropped to an 80×80 pixel window centered on the cell.

### Focus Analysis Results

The system analyzes 16 Z-stack images and selects the best-focused one:

**Best Focused Image**: `Image__2025-12-30__16-15-19.jpg` (Hybrid Score: 0.998)

Sample focus metrics for the Z-stack:

| Image | Laplacian | Tennengrad | Brenner | Hybrid Score |
|-------|-----------|------------|---------|--------------|
| Image__2025-12-30__16-14-35.jpg | 121.62 | 441.32 | 352778774 | 0.011 |
| Image__2025-12-30__16-14-49.jpg | 122.09 | 447.21 | 354651672 | 0.023 |
| Image__2025-12-30__16-14-53.jpg | 122.12 | 447.21 | 355009820 | 0.024 |
| Image__2025-12-30__16-14-57.jpg | 122.27 | 463.30 | 359238356 | 0.037 |
| Image__2025-12-30__16-15-05.jpg | 122.24 | 461.12 | 358659136 | 0.035 |
| ... | ... | ... | ... | ... |
| **Image__2025-12-30__16-15-19.jpg** | **highest** | **highest** | **highest** | **0.998** |

The hybrid score combines: 40% Laplacian + 40% Tennengrad + 20% Brenner.

**Weighting Rationale:**
- **Laplacian (40%)**: Second-order derivative operator, highly sensitive to edges and fine details in cellular structures
- **Tennengrad (40%)**: Sobel-based gradient measure, captures directional edge information, complements Laplacian's approach
- **Brenner (20%)**: Simpler pixel-difference metric, less weight due to higher noise sensitivity, serves as validation for gradient-based methods
- Gradient-based methods (Laplacian + Tennengrad = 80%) dominate because white blood cells exhibit strong boundaries that these operators detect reliably in microscopy images

## Sample Results

### Focus Analysis Visualization

The system generates focus measure curves to visualize image quality across the Z-stack:

![Focus Measure Curves](results-main/samples/1212/focus_measure_curves.png)
*Normalized focus metrics (Laplacian, Tennengrad, Brenner) across 16 Z-stack images*

![Hybrid Focus Curve](results-main/samples/1212/hybrid_focus_curve.png)
*Combined hybrid focus score - peak indicates the best-focused image*

### Cell Detection Pipeline

Below shows the step-by-step processing for white blood cell detection:

| Step | Image | Description |
|------|-------|-------------|
| Original | ![Original](results-main/samples/1212/original_image.png) | Best focused image from Z-stack |
| Blue Dominance | ![Blue Dom](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/blue_dom.png) | Blue channel extraction: B - 0.8R - 0.2G |
| Thresholded | ![Threshold](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/otsu_thresholded_blue_channel.png) | Otsu's automatic thresholding |
| After Opening | ![Opening](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/after_opening.png) | Morphological opening (noise removal) |
| After Closing | ![Closing](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/after_closing.png) | Morphological closing (gap filling) |
| Overlayed Labels | ![Overlay](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/overlayed_labels.png) | Detected cells marked in red |
| Large WBCs | ![Large](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/large_white_blood_cells.png) | Filtered large cells (area: 200-2000) |
| Small WBCs | ![Small](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/small_white_blood_cells.png) | Filtered small cells (area: <200) |

### Cropped Cell Examples

Each detected cell is extracted as an 80×80 pixel patch:

**Large White Blood Cells:**

![Large Cell 1](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/cropped_large_cells/large_cell_1.png)
![Large Cell 2](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/cropped_large_cells/large_cell_2.png)
![Large Cell 3](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/cropped_large_cells/large_cell_3.png)

**Small White Blood Cells:**

![Small Cell 1](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/cropped_small_cells/small_cell_1.png)
![Small Cell 2](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/cropped_small_cells/small_cell_2.png)
![Small Cell 3](results-main/samples/1212/blur_blue_scale/wr_0.8_wg_0.2/cropped_small_cells/small_cell_3.png)



### Implementation Comparison

- **main.py** (Non-vectorized): ~5-6 minutes for complete analysis
- **main_vectorized.py** (Vectorized): ~20-30 seconds for complete analysis

The vectorized version achieves approximately **10-18× speed improvement** by utilizing NumPy's optimized array operations instead of nested Python loops. This dramatic performance gain comes from:

- Vectorized convolution operations
- Optimized morphological operations (erosion/dilation)
- Efficient array slicing and indexing

Both implementations produce identical results - the only difference is execution speed.

## Algorithm Details

### Focus Analysis

Focus quality is evaluated using a weighted combination of classical focus measures:

- Laplacian Variance (40%)
- Tennengrad Gradient Measure (40%)
- Brenner Focus Measure (20%)

The image with the highest combined score is selected as the best-focused image.

### White Blood Cell Detection Pipeline

1. Gaussian smoothing to reduce noise
2. Blue channel dominance extraction
3. Otsu thresholding for binarization
4. Morphological opening to remove noise
5. Morphological closing to fill gaps
6. Connected component labeling
7. Area-based filtering of cell regions
8. Cropping fixed-size regions around detected cell centroids


## Authors

**Cemal Özçelik** (M.Sc.)  
**Başak Karakaş** (Ph.D.)

Gebze Technical University  
Department of Computer Engineering  
Course: Digital Image Processing

**Advisor**: Habil Kalkan
