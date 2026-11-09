# Image-Based Cell Screening

Python image-analysis pipeline for **image-based screening of CAR-T–cancer cell interactions** in two-channel fluorescence microscopy. It segments CAR-T cells and cancer cells, detects their contact regions, and extracts quantitative image-based features of each cell–cell encounter.

The pipeline was developed as part of an engineering project at the Hebrew University of Jerusalem. In that project, CAR-T cells on an upper coverslip are brought into controlled contact with cancer cells on a glass-bottom dish, using a 3D-printed holder, a manual Z-axis actuator and 20 µm spacer beads. The images analyzed here come from that platform.


## What it does

For each sample (a pair of images from the same field of view):

1. Loads the CAR-T image (488 nm / green) and the cancer-cell image (647 nm / red). A bright-field image is optional.
2. Applies Gaussian background correction and mild CLAHE contrast enhancement.
3. Recovers cell territories with Otsu-based hysteresis thresholding.
4. Separates neighboring cells with marker-controlled watershed (distance transform + local maxima).
5. Classifies CAR-T–cancer cell pairs as **confirmed** or **possible** contacts. For confirmed contacts, it defines the contact ROI as the full overlap of the two cell masks.
6. Measures cell area, ROI area, overlap, cell-to-cell distance and background-corrected fluorescence intensity in both channels.
7. Optionally compares the automatic masks with manual masks (Precision, Recall, Dice, IoU).

## Repository structure

```
cell_analysis_pipeline.py   # main pipeline
cad/concepts/               # early OpenSCAD concept designs of the cell-contact device
data/images_folder/         # default input folder (images not included)
```

## Requirements

- Python 3.<x>
- numpy, pandas, matplotlib, scipy, scikit-image

```
pip install numpy pandas matplotlib scipy scikit-image
```

## Input files

Put all images in one folder. File names decide each image's role (the keywords are set at the top of the script):

- CAR-T channel: name contains `green`, `488`, `car`, `cart`, ...
- Cancer channel: name contains `red`, `647`, `cancer`, `target`, ...
- Bright-field (optional): name contains `bf` or `brightfield`

Example: `sample1_green.tif` and `sample1_red.tif`.

Optional manual masks for validation go in the same folder, named like `sample1_manual_car_t_mask.tif`, `sample1_manual_cancer_mask.tif`, `sample1_manual_contact_roi.tif`.

The microscopy images used in the project are not included in this repository.

## Usage

1. Open `cell_analysis_pipeline.py` and, in the `__main__` block at the bottom, set `input_folder` (default: `data/images_folder`) and `output_dir`.
2. Set `pixel_size_um` to your image calibration in µm per pixel. Example: 0.4 µm/pixel for a 40× objective with an Andor iXon Ultra 897 EMCCD camera (16 µm pixels).
3. Run:

```
python cell_analysis_pipeline.py
```

## Output

Written to `output_dir`:
- Cell-level and interaction-level measurements, and validation metrics (timestamped CSV files)
- Segmentation masks, contact-ROI masks and overlay images
- Processing-debug figures and per-sample summary graphs

## Validation status

In a preliminary comparison with manually annotated masks, the pipeline reached mean Dice ≈ 0.75, IoU ≈ 0.61, and precision and recall ≈ 0.76. The validation dataset is small, so these numbers are preliminary.

## Limitations

- A detected "contact" is an image-based spatial overlap. It does not by itself confirm immune-synapse formation or CAR-T activation.
- Closely packed or weakly fluorescent cells may occasionally remain merged after watershed separation.
- Segmentation quality depends on illumination uniformity and on the two channels being in the same field of view and focal plane.


# Authors
Yasmin Zahalka, Adan Hammam

Supervisors: Prof. Eilon Sherman, Dr. Julia Sajman

Racah Institute of Physics, The Hebrew University of Jerusalem

# Contact
- Yasmin Zahalka: Yasminzahalka27@gmail.com
- Adan Hammam: hammamadan@gmail.com
