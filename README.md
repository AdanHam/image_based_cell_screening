# Image-Based Cell Sorting

Python-based image-analysis pipeline for CAR-T cell screening using
fluorescence microscopy. The pipeline segments CAR-T cells and cancer
cells in two-channel fluorescence images, detects their contact regions,
and extracts quantitative image-based features to support the
characterization of cell–cell interactions.

# What it does
For each sample (a pair of images from the same field of view):
1. Loads the CAR-T image (488 nm / green) and the cancer-cell image
   (647 nm / red). A bright-field image is optional.
2. Applies background correction and mild CLAHE contrast enhancement.
3. Recovers cell territories with Otsu-based hysteresis thresholding.
4. Separates neighboring cells with marker-controlled watershed.
5. Classifies CAR-T–cancer cell pairs as confirmed or possible contacts
   and defines the contact-region ROI.
6. Measures cell area, ROI area, overlap, cell-to-cell distance and
   fluorescence intensity in both channels.
7. Optionally compares the automatic masks with manual masks
   (Precision, Recall, Dice, IoU).

# Requirements
- Python 3
- numpy, pandas, matplotlib, scipy, scikit-image

```
pip install numpy pandas matplotlib scipy scikit-image
```

# Input files
Put all images in one folder. File names decide each image's role
(keywords are set at the top of the script):
- CAR-T channel: name contains `green`, `488`, `car`, `cart`, ...
- Cancer channel: name contains `red`, `647`, `cancer`, `target`, ...
- Bright-field (optional): name contains `bf` or `brightfield`

Example: `sample1_green.tif` and `sample1_red.tif`.

Optional manual masks for validation, in the same folder, named like
`sample1_manual_car_t_mask.tif`, `sample1_manual_cancer_mask.tif`,
`sample1_manual_contact_roi.tif`.

# Usage
1. Open `cell_analysis_pipeline.py` and, in the `__main__` block at the
   bottom, set `input_folder` (default: `data/images_folder`) and
   `output_dir`.
2. Set `pixel_size_um` to your image calibration (µm per pixel).
3. Run:
```
python cell_analysis_pipeline.py
```

# Output
Written to `output_dir`: cell and interaction measurements and
validation metrics (CSV, timestamped), segmentation and overlay images,
contact-ROI masks, and summary graphs.

# Authors
Yasmin Zahalka, Adan Hammam

Supervisors: Prof. Eilon Sherman, Dr. Julia Sajman

Racah Institute of Physics, The Hebrew University of Jerusalem

# Contact
- Yasmin Zahalka: yasmin.zahalka@mail.huji.ac.il
- Adan Hammam: adan.hamam@mail.huji.ac.il
