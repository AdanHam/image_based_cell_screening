import os
import glob
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage import filters, measure, morphology, segmentation, feature, exposure
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    binary_dilation,
    binary_fill_holes,
)


# =========================================================
# VERSION NOTE (v9)
# =========================================================
# Adds dual-channel fluorescence quantification for every contact ROI:
#   488 / green = CAR-T channel
#   647 / red   = cancer-cell channel
# Existing ROI columns are preserved for backward compatibility and still
# refer to the 647/red channel. New roi_488_* and roi_647_* columns are
# written explicitly to the interaction CSV.


# =========================================================
# SETTINGS
# =========================================================

IMAGE_EXTENSIONS = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]

BF_KEYWORDS = ["bf", "brightfield", "bright-field", "bright_field"]
GREEN_KEYWORDS = ["green", "gfp", "car", "cart", "car-t", "tcell", "t-cell", "t_cell", "488"]
RED_KEYWORDS = ["red", "rfp", "cancer", "target", "tumor", "tumour", "647"]


# =========================================================
# 1. IMAGE LOADING AND GROUPING
# =========================================================

def list_image_files(input_folder):
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found in folder: {input_folder}")
    return image_paths


def load_grayscale_image(path):
    img = imread(path)
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3]
        return np.mean(img, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported image shape {img.shape} in file: {path}")


def detect_image_role(path):
    filename = os.path.basename(path).lower()

    if any(k in filename for k in ["manual", "mask", "weka", "groundtruth", "ground_truth"]):
        return "manual"
    if any(k in filename for k in BF_KEYWORDS):
        return "bf"
    if any(k in filename for k in GREEN_KEYWORDS):
        return "green"
    if any(k in filename for k in RED_KEYWORDS):
        return "red"
    return "unknown"


def get_sample_name(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    all_keywords = (
        BF_KEYWORDS + GREEN_KEYWORDS + RED_KEYWORDS
        + ["manual", "mask", "weka", "groundtruth", "ground_truth"]
    )
    for key in all_keywords:
        name = name.replace(key, "")
    while "__" in name:
        name = name.replace("__", "_")
    while "--" in name:
        name = name.replace("--", "-")
    name = name.strip("_- ")
    if not name:
        name = os.path.splitext(os.path.basename(path))[0]
    return name


def organize_grayscale_images(input_folder):
    image_paths = list_image_files(input_folder)
    groups = {}
    unknown_files = []

    for path in image_paths:
        role = detect_image_role(path)
        if role == "manual":
            continue
        sample_name = get_sample_name(path)
        if role == "unknown":
            unknown_files.append(path)
            continue

        groups.setdefault(sample_name, {
            "sample_name": sample_name,
            "bf_path": None,
            "green_path": None,
            "red_path": None,
        })

        if role == "bf":
            groups[sample_name]["bf_path"] = path
        elif role == "green":
            groups[sample_name]["green_path"] = path
        elif role == "red":
            groups[sample_name]["red_path"] = path

    valid_groups = [
        g for g in groups.values()
        if g["green_path"] is not None and g["red_path"] is not None
    ]

    if not valid_groups:
        print("Unknown files:")
        for p in unknown_files:
            print(" -", p)
        raise ValueError(
            "No valid image groups found. Each sample needs one green/488 image "
            "and one red/647 image. Example: sample1_green.tif and sample1_red.tif"
        )

    return valid_groups


# =========================================================
# 2. PREPROCESSING
# =========================================================

def normalize_image(img, low_percentile=1.0, high_percentile=99.5):
    img = img.astype(np.float32)
    low, high = np.nanpercentile(img, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - low) / (high - low)
    return np.clip(out, 0, 1).astype(np.float32)


def preprocess_frame(frame, smooth_sigma=1.0, background_sigma=35.0):
    """
    Large background_sigma is intentional.
    In the previous version background_sigma=8 could remove broad cell signal
    and leave only bright subcellular fragments.
    """
    frame = frame.astype(np.float32)
    smoothed = gaussian_filter(frame, sigma=smooth_sigma)
    background = gaussian_filter(smoothed, sigma=background_sigma)
    corrected = smoothed - background
    corrected[corrected < 0] = 0
    return normalize_image(corrected)


def get_threshold(processed, threshold_method):
    if threshold_method == "otsu":
        return filters.threshold_otsu(processed)
    if threshold_method == "yen":
        return filters.threshold_yen(processed)
    if threshold_method == "li":
        return filters.threshold_li(processed)
    if threshold_method == "triangle":
        return filters.threshold_triangle(processed)
    raise ValueError("threshold_method must be 'otsu', 'yen', 'li', or 'triangle'")


# =========================================================
# 3. IMPROVED SEGMENTATION
# =========================================================

def segment_cells(
    grayscale_image,
    min_size=80,
    max_size=None,
    min_distance=18,
    threshold_method="otsu",
    threshold_scale=0.85,
    smooth_sigma=1.0,
    background_sigma=35.0,
    closing_radius=2,
    opening_radius=0,
    hole_area=200,
    distance_smooth_sigma=2.0,
    marker_method="hmax",
    h_maxima_height=2.0,
    min_solidity=0.0,
    max_eccentricity=1.0,
):
    """
    Improved classical segmentation designed to reduce fragmentation and
    over-segmentation without requiring training data.

    Main changes compared with the previous version:
    1. Larger background Gaussian -> preserves broad cell bodies.
    2. threshold_scale < 1 -> includes dimmer parts of each fluorescent cell.
    3. Morphological closing -> reconnects small gaps inside one cell.
    4. Larger hole filling -> reconstructs cell interiors.
    5. Stronger smoothing of distance map.
    6. h-maxima markers -> suppresses weak local maxima that split one cell.
    7. Larger default min_distance and min_size.
    """

    processed = preprocess_frame(
        grayscale_image,
        smooth_sigma=smooth_sigma,
        background_sigma=background_sigma,
    )

    if processed.max() <= 0:
        empty = np.zeros_like(processed, dtype=np.int32)
        return empty, processed, empty.astype(bool)

    threshold_value = get_threshold(processed, threshold_method)
    effective_threshold = float(np.clip(threshold_value * threshold_scale, 0, 1))
    binary = processed > effective_threshold

    # Reconnect small gaps before object filtering.
    if closing_radius and closing_radius > 0:
        binary = morphology.closing(binary, morphology.disk(closing_radius))

    # Optional opening if salt-noise is severe. Default off to avoid shrinking cells.
    if opening_radius and opening_radius > 0:
        binary = morphology.opening(binary, morphology.disk(opening_radius))

    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=hole_area)

    if np.sum(binary) == 0:
        empty = np.zeros_like(processed, dtype=np.int32)
        return empty, processed, binary

    distance = distance_transform_edt(binary)
    distance_smooth = gaussian_filter(distance, sigma=distance_smooth_sigma)

    # Robust marker detection. h-maxima is less likely than raw local maxima
    # to generate multiple markers inside a single fluorescent cell.
    if marker_method == "hmax":
        h = max(float(h_maxima_height), 0.1)
        marker_mask = morphology.h_maxima(distance_smooth, h=h)
        marker_mask &= binary
        markers = measure.label(marker_mask)

        # If h-maxima is too strict, fall back to peak_local_max.
        if markers.max() == 0:
            peak_coords = feature.peak_local_max(
                distance_smooth,
                min_distance=min_distance,
                labels=binary,
            )
            markers = np.zeros_like(processed, dtype=np.int32)
            for marker_id, (row, col) in enumerate(peak_coords, start=1):
                markers[row, col] = marker_id
    elif marker_method == "peaks":
        peak_coords = feature.peak_local_max(
            distance_smooth,
            min_distance=min_distance,
            labels=binary,
        )
        markers = np.zeros_like(processed, dtype=np.int32)
        for marker_id, (row, col) in enumerate(peak_coords, start=1):
            markers[row, col] = marker_id
    else:
        raise ValueError("marker_method must be 'hmax' or 'peaks'")

    # If no useful markers were found, keep connected components rather than failing.
    if markers.max() == 0:
        labeled = measure.label(binary)
    else:
        labeled = segmentation.watershed(
            -distance_smooth,
            markers,
            mask=binary,
        )

    clean_labeled = np.zeros_like(labeled, dtype=np.int32)
    new_label = 1

    for region in measure.regionprops(labeled):
        if region.area < min_size:
            continue
        if max_size is not None and region.area > max_size:
            continue
        if region.solidity < min_solidity:
            continue
        if region.eccentricity > max_eccentricity:
            continue

        clean_labeled[labeled == region.label] = new_label
        new_label += 1

    final_binary = clean_labeled > 0
    return clean_labeled, processed, final_binary


# =========================================================
# 3B. CAR-T INSTANCE SEGMENTATION
# =========================================================

def _ensure_marker_per_component(markers, geometry_mask, distance_smooth):
    """Make sure every connected cell territory has at least one watershed marker."""
    components = measure.label(geometry_mask)
    next_label = int(markers.max()) + 1

    for region in measure.regionprops(components):
        component_mask = components == region.label
        if np.any(markers[component_mask] > 0):
            continue

        coords = np.argwhere(component_mask)
        if coords.size == 0:
            continue

        vals = distance_smooth[component_mask]
        best = coords[int(np.argmax(vals))]
        markers[best[0], best[1]] = next_label
        next_label += 1

    return markers


def _add_secondary_markers_for_suspicious_components(
    markers,
    geometry_mask,
    distance_smooth,
    primary_min_distance=12,
    secondary_min_distance=7,
    merged_area_factor=1.8,
):
    """
    Add extra markers only to unusually large connected territories.

    This protects against the specific failure mode where two adjacent CAR-T
    cells are joined by faint fluorescence and would otherwise become one cell.
    We do NOT split every large-looking object blindly. The object must be
    substantially larger than the median connected territory before a second,
    more permissive peak search is attempted.
    """
    components = measure.label(geometry_mask)
    props = measure.regionprops(components)
    if len(props) < 2:
        return markers

    areas = np.array([p.area for p in props], dtype=float)
    median_area = float(np.median(areas))
    if median_area <= 0:
        return markers

    next_label = int(markers.max()) + 1

    for region in props:
        if region.area < merged_area_factor * median_area:
            continue

        comp_mask = components == region.label
        existing_coords = np.argwhere((markers > 0) & comp_mask)

        # Search for additional centers with a smaller separation only inside
        # suspiciously large connected objects.
        extra_peaks = feature.peak_local_max(
            distance_smooth,
            min_distance=max(2, int(secondary_min_distance)),
            threshold_abs=1.0,
            labels=comp_mask,
            exclude_border=False,
        )

        for row, col in extra_peaks:
            # Keep a new peak only if it is not too close to an existing marker.
            if existing_coords.size:
                d = np.sqrt(
                    (existing_coords[:, 0] - row) ** 2
                    + (existing_coords[:, 1] - col) ** 2
                )
                if np.min(d) < max(3, primary_min_distance * 0.55):
                    continue

            markers[row, col] = next_label
            next_label += 1
            existing_coords = np.vstack([existing_coords, [row, col]]) if existing_coords.size else np.array([[row, col]])

    return markers



def _resplit_large_final_instances(
    labeled,
    min_size,
    area_factor=2.5,
    min_peak_distance=8,
    min_daughter_fraction=0.30,
    max_parts=4,
    distance_smooth_sigma=1.0,
):
    """
    Conservatively re-split only final instances that are much larger than the
    median instance area.

    Why this exists:
        Seeded watershed can still leave two crowded cells as one object when
        their geometric territory contains only one strong primary center.
        This second pass is deliberately limited to unusually large instances.

    A proposed split is accepted only when:
        - the parent area is >= area_factor * median final-instance area,
        - at least two internal distance peaks are found,
        - no daughter is smaller than max(min_size, min_daughter_fraction*median).

    Otherwise the original instance is kept unchanged. This makes the step
    conservative and avoids splitting ordinary single cells just because their
    fluorescence is non-uniform.
    """
    props = measure.regionprops(labeled)
    if len(props) < 3:
        return labeled

    areas = np.array([p.area for p in props if p.area >= min_size], dtype=float)
    if len(areas) < 3:
        return labeled

    median_area = float(np.median(areas))
    if median_area <= 0:
        return labeled

    out = np.zeros_like(labeled, dtype=np.int32)
    next_label = 1
    min_daughter_area = max(int(min_size), int(np.ceil(min_daughter_fraction * median_area)))

    for region in props:
        region_mask = labeled == region.label

        # Ordinary instances are copied unchanged.
        if region.area < area_factor * median_area:
            out[region_mask] = next_label
            next_label += 1
            continue

        local_distance = distance_transform_edt(region_mask)
        local_distance_smooth = gaussian_filter(
            local_distance,
            sigma=distance_smooth_sigma,
        )

        peaks = feature.peak_local_max(
            local_distance_smooth,
            min_distance=max(2, int(min_peak_distance)),
            threshold_abs=1.0,
            labels=region_mask,
            exclude_border=False,
        )

        if len(peaks) < 2:
            out[region_mask] = next_label
            next_label += 1
            continue

        # Do not allow a huge number of artificial daughters. The expected
        # number is estimated from parent_area / median_area and capped.
        expected_parts = int(np.clip(np.round(region.area / median_area), 2, max_parts))
        peak_values = np.array([local_distance_smooth[r, c] for r, c in peaks])
        order = np.argsort(peak_values)[::-1][:expected_parts]
        peaks = peaks[order]

        local_markers = np.zeros_like(labeled, dtype=np.int32)
        for marker_id, (row, col) in enumerate(peaks, start=1):
            local_markers[row, col] = marker_id

        local_split = segmentation.watershed(
            -local_distance_smooth,
            local_markers,
            mask=region_mask,
        )
        daughters = measure.regionprops(local_split)
        daughter_areas = [d.area for d in daughters]

        # Accept only a clean split. If one daughter would be a tiny fragment,
        # the parent is safer to keep as one instance.
        if len(daughters) < 2 or min(daughter_areas) < min_daughter_area:
            out[region_mask] = next_label
            next_label += 1
            continue

        for daughter in daughters:
            out[local_split == daughter.label] = next_label
            next_label += 1

    return out

def segment_seeded_instances(
    grayscale_image,
    min_size=80,
    max_size=None,
    threshold_method="otsu",
    smooth_sigma=1.0,
    background_sigma=35.0,
    clahe_clip_limit=0.010,
    clahe_kernel_size=None,
    low_threshold_scale=0.55,
    high_threshold_scale=0.95,
    closing_radius=1,
    geometry_hole_area=1200,
    distance_smooth_sigma=1.5,
    marker_min_distance=12,
    secondary_marker_min_distance=7,
    merged_area_factor=1.8,
    enable_large_instance_resplit=True,
    resplit_area_factor=2.5,
    resplit_min_peak_distance=8,
    resplit_min_daughter_fraction=0.30,
    resplit_max_parts=4,
    min_solidity=0.0,
    max_eccentricity=1.0,
):
    """
    Seeded instance segmentation for faint borders AND crowded cells.

    Key idea
    --------
    A single global threshold can be poor for fluorescence channels when bright
    cells dominate the image while dim cells and dim cell edges disappear.
    Lowering the threshold alone is also unsafe because neighboring cells can
    merge. This function therefore separates two tasks:

    1. Recover the full possible cell territory using mild CLAHE + hysteresis.
    2. Split touching territories into individual cells using marker-controlled
       watershed on a filled geometric mask.

    The low threshold can therefore recover faint borders without forcing two
    adjacent cells to become one final instance.

    Returns
    -------
    labeled : int image
        Final instance labels; every detected cell has a separate integer label.
    processed : float image
        Background-corrected image before CLAHE.
    geometry_mask : bool image
        Full cell territory used for area, distance and contact calculations.
    debug : dict
        Intermediate masks/images for visual inspection.
    """
    processed = preprocess_frame(
        grayscale_image,
        smooth_sigma=smooth_sigma,
        background_sigma=background_sigma,
    )

    if processed.max() <= 0:
        empty_i = np.zeros_like(processed, dtype=np.int32)
        empty_b = np.zeros_like(processed, dtype=bool)
        return empty_i, processed, empty_b, {
            "enhanced": processed,
            "strong_seeds": empty_b,
            "hysteresis": empty_b,
            "geometry": empty_b,
            "markers": empty_i,
        }

    # Mild local contrast enhancement. The clip limit is deliberately small:
    # enough to recover dim cells, but not so strong that background texture
    # becomes cell-like.
    enhanced = exposure.equalize_adapthist(
        processed,
        kernel_size=clahe_kernel_size,
        clip_limit=clahe_clip_limit,
        nbins=256,
    ).astype(np.float32)

    base_threshold = get_threshold(enhanced, threshold_method)
    low_thr = float(np.clip(base_threshold * low_threshold_scale, 0.0, 1.0))
    high_thr = float(np.clip(base_threshold * high_threshold_scale, 0.0, 1.0))

    if high_thr <= low_thr:
        high_thr = min(1.0, low_thr + 0.02)

    # Strong pixels are used only as confidence seeds. Faint pixels are accepted
    # if they are connected to a stronger region, which recovers cell edges more
    # safely than simply lowering one global threshold.
    strong_seeds = enhanced >= high_thr
    hysteresis = filters.apply_hysteresis_threshold(
        enhanced,
        low_thr,
        high_thr,
    )

    # Morphology is intentionally gentle. Large closing kernels can bridge two
    # neighboring cells, so the default radius is only one pixel.
    territory = hysteresis.copy()
    if closing_radius and closing_radius > 0:
        territory = morphology.binary_closing(
            territory,
            footprint=morphology.disk(closing_radius),
        )

    territory = morphology.remove_small_objects(
        territory,
        min_size=min_size,
    )

    # Fill internal dark holes for GEOMETRY only. This does not connect separate
    # objects; it merely makes ring-shaped cells solid for distance-transform
    # based center detection and contact geometry.
    geometry_mask = binary_fill_holes(territory).astype(bool)
    if geometry_hole_area is not None and geometry_hole_area > 0:
        geometry_mask = morphology.remove_small_holes(
            geometry_mask,
            area_threshold=int(geometry_hole_area),
        )

    geometry_mask = morphology.remove_small_objects(
        geometry_mask,
        min_size=min_size,
    )

    if not np.any(geometry_mask):
        empty_i = np.zeros_like(processed, dtype=np.int32)
        return empty_i, processed, geometry_mask, {
            "enhanced": enhanced,
            "strong_seeds": strong_seeds,
            "hysteresis": hysteresis,
            "geometry": geometry_mask,
            "markers": empty_i,
        }

    # Distance-transform peaks represent candidate CELL CENTERS, not bright
    # fluorescence spots. This is the important part that keeps adjacent cells
    # as separate instances even if their faint borders touch.
    distance = distance_transform_edt(geometry_mask)
    distance_smooth = gaussian_filter(
        distance,
        sigma=distance_smooth_sigma,
    )

    peak_coords = feature.peak_local_max(
        distance_smooth,
        min_distance=max(2, int(marker_min_distance)),
        threshold_abs=1.0,
        labels=geometry_mask,
        exclude_border=False,
    )

    markers = np.zeros_like(processed, dtype=np.int32)
    for marker_id, (row, col) in enumerate(peak_coords, start=1):
        markers[row, col] = marker_id

    markers = _ensure_marker_per_component(
        markers,
        geometry_mask,
        distance_smooth,
    )

    # If one connected territory is much larger than the typical object, make
    # one additional, more permissive center search inside that territory.
    markers = _add_secondary_markers_for_suspicious_components(
        markers,
        geometry_mask,
        distance_smooth,
        primary_min_distance=marker_min_distance,
        secondary_min_distance=secondary_marker_min_distance,
        merged_area_factor=merged_area_factor,
    )

    labeled = segmentation.watershed(
        -distance_smooth,
        markers,
        mask=geometry_mask,
    )

    # v7: conservative second-pass split for unusually large final instances.
    # This is different from the earlier marker enrichment step: it evaluates
    # the *actual first-pass instances* and only revisits strong outliers.
    if enable_large_instance_resplit:
        labeled = _resplit_large_final_instances(
            labeled=labeled,
            min_size=min_size,
            area_factor=resplit_area_factor,
            min_peak_distance=resplit_min_peak_distance,
            min_daughter_fraction=resplit_min_daughter_fraction,
            max_parts=resplit_max_parts,
            distance_smooth_sigma=max(0.5, distance_smooth_sigma * 0.75),
        )

    clean_labeled = np.zeros_like(labeled, dtype=np.int32)
    new_label = 1
    for region in measure.regionprops(labeled):
        if region.area < min_size:
            continue
        if max_size is not None and region.area > max_size:
            continue
        if region.solidity < min_solidity:
            continue
        if region.eccentricity > max_eccentricity:
            continue

        clean_labeled[labeled == region.label] = new_label
        new_label += 1

    geometry_mask = clean_labeled > 0

    debug = {
        "enhanced": enhanced,
        "strong_seeds": strong_seeds,
        "hysteresis": hysteresis,
        "geometry": geometry_mask,
        "markers": markers,
        "distance": distance_smooth,
        "low_threshold": low_thr,
        "high_threshold": high_thr,
    }

    return clean_labeled, processed, geometry_mask, debug


# =========================================================
# 4. CELL MEASUREMENTS
# =========================================================

def extract_cells(
    labeled_image,
    intensity_image,
    cell_type,
    background_ring_px=5,
    pixel_size_um=0.108,
):
    cells = []
    props = measure.regionprops(labeled_image, intensity_image=intensity_image)
    all_cells_mask = labeled_image > 0

    for p in props:
        coords = p.coords
        area_px = int(p.area)
        area_um2 = area_px * (pixel_size_um ** 2)

        cell_pixels = intensity_image[coords[:, 0], coords[:, 1]]
        raw_mean = float(np.mean(cell_pixels)) if cell_pixels.size else np.nan
        raw_total = float(np.sum(cell_pixels)) if cell_pixels.size else np.nan

        cell_mask = labeled_image == p.label
        dilated_mask = binary_dilation(cell_mask, iterations=background_ring_px)
        background_ring = dilated_mask & ~cell_mask & ~all_cells_mask
        background_pixels = intensity_image[background_ring]
        local_bg = float(np.mean(background_pixels)) if background_pixels.size else 0.0

        corrected_total = raw_total - area_px * local_bg if not np.isnan(raw_total) else np.nan
        corrected_mean = corrected_total / area_px if area_px > 0 else np.nan
        centroid_y, centroid_x = p.centroid

        cells.append({
            "cell_id": int(p.label),
            "cell_type": cell_type,
            "area_px": area_px,
            "area_um2": float(area_um2),
            "centroid_y": float(centroid_y),
            "centroid_x": float(centroid_x),
            "mean_intensity": raw_mean,
            "total_intensity": raw_total,
            "integrated_density": raw_total,
            "local_background_mean": local_bg,
            "corrected_mean_intensity": corrected_mean,
            "corrected_total_intensity": corrected_total,
            "corrected_integrated_density": corrected_total,
            "coords": coords,
        })

    return cells


def annotate_cell_quality(
    cells,
    small_area_factor=0.35,
    large_area_factor=2.5,
    image_shape=None,
):
    """
    Add non-destructive quality flags to detected cells.

    Nothing is removed from the analysis. The flags are written to the CSV so
    suspicious small objects, possible merged objects, and border-truncated
    objects can be inspected separately later.
    """
    if not cells:
        return cells

    areas = np.array([c["area_px"] for c in cells], dtype=float)
    median_area = float(np.median(areas)) if len(areas) else np.nan

    for cell in cells:
        coords = cell["coords"]
        touches_border = False
        if image_shape is not None and len(coords):
            h, w = image_shape
            touches_border = bool(
                np.any(coords[:, 0] == 0)
                or np.any(coords[:, 0] == h - 1)
                or np.any(coords[:, 1] == 0)
                or np.any(coords[:, 1] == w - 1)
            )

        ratio = cell["area_px"] / median_area if median_area > 0 else np.nan

        if touches_border:
            quality = "border_object"
        elif np.isfinite(ratio) and ratio < small_area_factor:
            quality = "possible_small_object"
        elif np.isfinite(ratio) and ratio > large_area_factor:
            quality = "possible_merged"
        else:
            quality = "valid"

        cell["touches_border"] = touches_border
        cell["area_ratio_to_median"] = float(ratio) if np.isfinite(ratio) else np.nan
        cell["cell_quality"] = quality

    return cells


# =========================================================
# 5. CONTACT DETECTION
# =========================================================

def create_cell_mask(coords, image_shape):
    mask = np.zeros(image_shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def min_distance_between_cells(coords1, coords2, image_shape):
    if len(coords1) == 0 or len(coords2) == 0:
        return np.nan
    mask1 = create_cell_mask(coords1, image_shape)
    mask2 = create_cell_mask(coords2, image_shape)
    distance_map = distance_transform_edt(~mask1)
    return float(np.min(distance_map[mask2]))


def max_radius_from_centroid(cell):
    """
    Maximum pixel distance from the cell centroid to any pixel in the cell.

    This is used only as a SAFE pre-filter before the more accurate
    mask-to-mask distance calculation. Unlike a fixed centroid cutoff,
    this automatically adapts to large and irregular cells.

    If two cells could be within possible_distance_px of one another, then
    their centroid distance cannot exceed:

        radius_cell_1 + radius_cell_2 + possible_distance_px

    Therefore a true/possible contact will not be discarded simply because
    the cell centroids are more than an arbitrary fixed number of pixels apart.
    """
    coords = cell.get("coords")
    if coords is None or len(coords) == 0:
        return 0.0

    dy = coords[:, 0] - cell["centroid_y"]
    dx = coords[:, 1] - cell["centroid_x"]
    return float(np.max(np.hypot(dx, dy)))


def is_geometrically_possible_pair(
    car_t_cell,
    cancer_cell,
    centroid_distance_px,
    possible_distance_px,
    safety_margin_px=2,
):
    """
    Safe adaptive candidate-pair test.

    v5 used a fixed rule:
        centroid_distance_px <= 40

    That can incorrectly reject touching large cells whose centroids are
    farther apart than 40 px. v6 uses each segmented cell's actual spatial
    extent instead.
    """
    r_car_t = max_radius_from_centroid(car_t_cell)
    r_cancer = max_radius_from_centroid(cancer_cell)

    adaptive_limit_px = (
        r_car_t
        + r_cancer
        + possible_distance_px
        + safety_margin_px
    )

    return centroid_distance_px <= adaptive_limit_px, adaptive_limit_px


def compute_contact_roi(
    car_t_cell,
    cancer_cell,
    image_shape,
    roi_dilation_px=2,
    overlap_expand_px=0,
):
    """
    Build a continuous ROI representing the full spatial contact region.

    Previous behavior:
        The ROI was the intersection of two *dilated boundaries*.
        When the two segmented cells overlap substantially, their outer
        boundaries meet only near the ends of the overlap. This produces
        a few isolated yellow points instead of the full contact area.

    New behavior:
        1. If the two cell masks overlap, the complete overlap region is
           used as the contact ROI. This gives one continuous area covering
           the whole shared region between the two segmented cells.
        2. If there is no direct overlap, a thin continuous interface zone
           is constructed from the overlap of dilated whole-cell masks.

    Parameters
    ----------
    roi_dilation_px : int
        Used only when the masks do not overlap. It controls how far each
        mask is expanded in order to define a near-contact interface.

    overlap_expand_px : int
        Optional small expansion of the true overlap ROI. Keep this at 0
        when you want the ROI to be exactly the shared mask area.
    """

    car_t_mask = create_cell_mask(car_t_cell["coords"], image_shape)
    cancer_mask = create_cell_mask(cancer_cell["coords"], image_shape)

    # -----------------------------------------------------
    # Case 1: true mask overlap -> use the ENTIRE overlap.
    # -----------------------------------------------------
    overlap_mask = car_t_mask & cancer_mask

    if np.any(overlap_mask):
        roi_mask = overlap_mask.copy()

        # Fill tiny holes and close one-pixel breaks so the ROI remains
        # continuous even when the segmentation boundary is slightly noisy.
        roi_mask = morphology.binary_closing(
            roi_mask,
            footprint=morphology.disk(1)
        )
        roi_mask = morphology.remove_small_holes(
            roi_mask,
            area_threshold=16
        )

        # Optional expansion while remaining inside the union of both cells.
        if overlap_expand_px > 0:
            roi_mask = binary_dilation(
                roi_mask,
                iterations=overlap_expand_px
            )
            roi_mask &= (car_t_mask | cancer_mask)

        return roi_mask

    # -----------------------------------------------------
    # Case 2: no overlap -> use the complete near-contact
    # interface created by expanding the whole cell masks.
    # -----------------------------------------------------
    car_t_zone = binary_dilation(
        car_t_mask,
        iterations=roi_dilation_px
    )
    cancer_zone = binary_dilation(
        cancer_mask,
        iterations=roi_dilation_px
    )

    roi_mask = car_t_zone & cancer_zone

    if np.any(roi_mask):
        roi_mask = morphology.binary_closing(
            roi_mask,
            footprint=morphology.disk(1)
        )
        roi_mask = morphology.remove_small_holes(
            roi_mask,
            area_threshold=16
        )

    return roi_mask


def classify_contact(
    car_t_cell,
    cancer_cell,
    image_shape,
    response_image_red,
    response_image_green,
    contact_distance_px=2,
    possible_distance_px=8,
    roi_dilation_px=2,
    pixel_size_um=0.108,
    roi_background_ring_px=5,
    min_contact_overlap_px=25,
    min_contact_overlap_fraction=0.02,
):
    """
    Conservative contact classification (v7).

    Definite contact requires a meaningful overlap between the two segmented
    cell masks. A very small overlap or a short gap is classified only as
    possible_contact, because both can be caused by segmentation uncertainty or
    small channel-registration errors.

    This prevents a 1-2 pixel near miss from being reported as a confirmed
    biological interaction.
    """
    car_t_mask = create_cell_mask(car_t_cell["coords"], image_shape)
    cancer_mask = create_cell_mask(cancer_cell["coords"], image_shape)

    overlap_mask = car_t_mask & cancer_mask
    overlap_area_px = int(np.sum(overlap_mask))
    overlap_area_um2 = overlap_area_px * (pixel_size_um ** 2)

    smaller_cell_area_px = max(1, min(car_t_cell["area_px"], cancer_cell["area_px"]))
    overlap_fraction_smaller_cell = overlap_area_px / smaller_cell_area_px
    required_overlap_px = max(
        int(min_contact_overlap_px),
        int(np.ceil(min_contact_overlap_fraction * smaller_cell_area_px)),
    )

    min_distance_px = min_distance_between_cells(
        car_t_cell["coords"], cancer_cell["coords"], image_shape
    )
    min_distance_um = min_distance_px * pixel_size_um if not np.isnan(min_distance_px) else np.nan

    roi_mask = compute_contact_roi(
        car_t_cell,
        cancer_cell,
        image_shape=image_shape,
        roi_dilation_px=roi_dilation_px,
        overlap_expand_px=0,
    )
    roi_area_px = int(np.sum(roi_mask))
    roi_area_um2 = roi_area_px * (pixel_size_um ** 2)

    # v7 classification hierarchy:
    # 1) meaningful overlap -> definite contact
    # 2) tiny overlap -> possible contact
    # 3) no overlap but close boundaries -> possible contact
    # 4) otherwise -> no contact
    if overlap_area_px >= required_overlap_px:
        contact_status = "contact_happened"
        contact_reason = "significant_overlap"
    elif overlap_area_px > 0:
        contact_status = "possible_contact"
        contact_reason = "small_overlap"
    elif min_distance_px <= possible_distance_px:
        contact_status = "possible_contact"
        contact_reason = (
            "very_close_without_overlap"
            if min_distance_px <= contact_distance_px
            else "near_without_overlap"
        )
    else:
        contact_status = "no_contact"
        contact_reason = "separated"

    # ---------------------------------------------------------
    # ROI fluorescence is quantified independently in BOTH channels:
    #   488 / green -> CAR-T signal
    #   647 / red   -> cancer-cell signal
    # The SAME geometric ROI and the SAME outside-cell background ring
    # are used for both channels, so the two fluorescence measurements
    # are directly comparable spatially.
    # ---------------------------------------------------------
    pair_union = car_t_mask | cancer_mask
    if roi_area_px > 0:
        pair_bg_outer = binary_dilation(
            pair_union,
            iterations=roi_background_ring_px
        )
        roi_bg_ring = pair_bg_outer & ~pair_union
    else:
        roi_bg_ring = np.zeros(image_shape, dtype=bool)

    def measure_roi_channel(channel_image):
        if roi_area_px <= 0:
            return {
                "background_mean": np.nan,
                "mean_intensity": np.nan,
                "total_intensity": np.nan,
                "integrated_density": np.nan,
                "max_intensity": np.nan,
                "corrected_mean_intensity": np.nan,
                "corrected_total_intensity": np.nan,
                "corrected_integrated_density": np.nan,
            }

        roi_pixels = channel_image[roi_mask]
        bg_pixels = channel_image[roi_bg_ring]

        background_mean = float(np.mean(bg_pixels)) if bg_pixels.size else np.nan
        mean_intensity = float(np.mean(roi_pixels))
        total_intensity = float(np.sum(roi_pixels))
        max_intensity = float(np.max(roi_pixels))

        if np.isnan(background_mean):
            corrected_total = np.nan
            corrected_mean = np.nan
        else:
            corrected_total = total_intensity - roi_area_px * background_mean
            corrected_mean = corrected_total / roi_area_px

        return {
            "background_mean": background_mean,
            "mean_intensity": mean_intensity,
            "total_intensity": total_intensity,
            "integrated_density": total_intensity,
            "max_intensity": max_intensity,
            "corrected_mean_intensity": corrected_mean,
            "corrected_total_intensity": corrected_total,
            "corrected_integrated_density": corrected_total,
        }

    roi_647 = measure_roi_channel(response_image_red)
    roi_488 = measure_roi_channel(response_image_green)
    # Dual-channel ROI measurement: the same ROI is also measured on the
    # sum of the 488-nm and 647-nm channels so that the default ROI metrics
    # represent both channels together rather than the red channel alone.
    combined_response_image = response_image_green + response_image_red
    roi_combined = measure_roi_channel(combined_response_image)

    return {
        "contact_status": contact_status,
        "contact_reason": contact_reason,
        "contact_happened": contact_status == "contact_happened",
        "possible_contact": contact_status == "possible_contact",
        "min_distance_px": min_distance_px,
        "min_distance_um": float(min_distance_um),
        "overlap_area_px": overlap_area_px,
        "overlap_area_um2": float(overlap_area_um2),
        "overlap_fraction_smaller_cell": float(overlap_fraction_smaller_cell),
        "required_overlap_px": int(required_overlap_px),
        "roi_area_px": roi_area_px,
        "roi_area_um2": float(roi_area_um2),
        # ROI fluorescence measurements on the combined 488+647 image.
        "roi_mean_intensity": roi_combined["mean_intensity"],
        "roi_total_intensity": roi_combined["total_intensity"],
        "roi_integrated_density": roi_combined["integrated_density"],
        "roi_max_intensity": roi_combined["max_intensity"],
        "roi_background_mean": roi_combined["background_mean"],
        "roi_corrected_mean_intensity": roi_combined["corrected_mean_intensity"],
        "roi_corrected_total_intensity": roi_combined["corrected_total_intensity"],
        "roi_corrected_integrated_density": roi_combined["corrected_integrated_density"],
        "roi_mask": roi_mask,
    }


# =========================================================
# 6. OPTIONAL MANUAL VALIDATION
# =========================================================

def load_manual_binary_mask(path):
    mask = imread(path)
    if mask.ndim == 3:
        mask = np.mean(mask[..., :3], axis=-1)
    return mask > 0


def compute_binary_validation_metrics(auto_mask, manual_mask):
    auto_mask = auto_mask.astype(bool)
    manual_mask = manual_mask.astype(bool)

    if auto_mask.shape != manual_mask.shape:
        raise ValueError(f"Mask shapes do not match: auto {auto_mask.shape}, manual {manual_mask.shape}")

    tp = int(np.sum(auto_mask & manual_mask))
    fp = int(np.sum(auto_mask & ~manual_mask))
    fn = int(np.sum(~auto_mask & manual_mask))
    tn = int(np.sum(~auto_mask & ~manual_mask))

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (tp + fn) if (tp + fn) else np.nan
    union = int(np.sum(auto_mask | manual_mask))
    iou = tp / union if union else np.nan
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": float(precision), "recall": float(recall),
        "false_negative_rate": float(fnr), "false_positive_rate": float(fpr),
        "iou": float(iou), "dice": float(dice),
    }



def compute_overall_validation_summary(validation_df):
    """
    Aggregate validation over every manual mask that was found.

    Two complementary summaries are reported:
    1) Macro metrics: every validated sample/mask receives equal weight.
    2) Pooled metrics: TP/FP/FN/TN are summed first, then the metrics are
       recomputed. This weights larger masks/images more strongly.

    The recommended single "success score" is macro Dice because Dice measures
    foreground-mask agreement without being dominated by the large background.
    """
    if validation_df is None or len(validation_df) == 0:
        return pd.DataFrame()

    required_count_cols = ["TP", "FP", "FN", "TN"]
    if not all(col in validation_df.columns for col in required_count_cols):
        raise ValueError(
            "Validation dataframe is missing TP/FP/FN/TN columns required "
            "for overall validation."
        )

    metric_cols = [
        m for m in ["precision", "recall", "iou", "dice"]
        if m in validation_df.columns
    ]

    rows = []

    def summarize_group(group_df, validation_type, aggregation_scope):
        tp = int(group_df["TP"].sum())
        fp = int(group_df["FP"].sum())
        fn = int(group_df["FN"].sum())
        tn = int(group_df["TN"].sum())

        pooled_precision = tp / (tp + fp) if (tp + fp) else np.nan
        pooled_recall = tp / (tp + fn) if (tp + fn) else np.nan
        pooled_iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
        pooled_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan

        macro = {
            metric: float(group_df[metric].mean(skipna=True))
            for metric in metric_cols
        }
        macro_std = {
            metric: float(group_df[metric].std(skipna=True, ddof=0))
            for metric in metric_cols
        }

        macro_dice = macro.get("dice", np.nan)

        return {
            "aggregation_scope": aggregation_scope,
            "validation_type": validation_type,
            "n_validated_masks": int(len(group_df)),
            "n_samples": int(group_df["sample_name"].nunique())
                if "sample_name" in group_df.columns else np.nan,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "macro_precision": macro.get("precision", np.nan),
            "macro_recall": macro.get("recall", np.nan),
            "macro_iou": macro.get("iou", np.nan),
            "macro_dice": macro_dice,
            "macro_precision_std": macro_std.get("precision", np.nan),
            "macro_recall_std": macro_std.get("recall", np.nan),
            "macro_iou_std": macro_std.get("iou", np.nan),
            "macro_dice_std": macro_std.get("dice", np.nan),
            "pooled_precision": float(pooled_precision),
            "pooled_recall": float(pooled_recall),
            "pooled_iou": float(pooled_iou),
            "pooled_dice": float(pooled_dice),
            # Recommended single-number success indicator.
            "success_score_percent": float(macro_dice * 100.0)
                if not np.isnan(macro_dice) else np.nan,
        }

    # One aggregate row per validation task across all available samples.
    for validation_type, group_df in validation_df.groupby("validation_type", sort=False):
        rows.append(
            summarize_group(
                group_df,
                validation_type=validation_type,
                aggregation_scope="all_samples_by_validation_type",
            )
        )

    # One grand aggregate row across every available manual validation mask.
    rows.append(
        summarize_group(
            validation_df,
            validation_type="ALL VALIDATION MASKS",
            aggregation_scope="all_samples_all_validation_types",
        )
    )

    return pd.DataFrame(rows)


def find_manual_mask_paths(input_folder, sample_name):
    exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    car_t = cancer = contact = None

    for ext in exts:
        candidates = {
            "car_t": [
                f"{sample_name}_manual_car_t_mask{ext}",
                f"{sample_name}_manual_cart_mask{ext}",
                f"{sample_name}_weka_car_t_mask{ext}",
                f"{sample_name}_groundtruth_car_t_mask{ext}",
            ],
            "cancer": [
                f"{sample_name}_manual_cancer_mask{ext}",
                f"{sample_name}_manual_target_mask{ext}",
                f"{sample_name}_weka_cancer_mask{ext}",
                f"{sample_name}_groundtruth_cancer_mask{ext}",
            ],
            "contact": [
                f"{sample_name}_manual_contact_roi{ext}",
                f"{sample_name}_manual_roi_mask{ext}",
                f"{sample_name}_weka_contact_roi{ext}",
                f"{sample_name}_groundtruth_contact_roi{ext}",
            ],
        }

        for filename in candidates["car_t"]:
            path = os.path.join(input_folder, filename)
            if os.path.exists(path):
                car_t = path
        for filename in candidates["cancer"]:
            path = os.path.join(input_folder, filename)
            if os.path.exists(path):
                cancer = path
        for filename in candidates["contact"]:
            path = os.path.join(input_folder, filename)
            if os.path.exists(path):
                contact = path

    return car_t, cancer, contact


def validate_sample(sample_name, input_folder, car_t_labeled, cancer_labeled, combined_roi_mask):
    rows = []
    car_t_path, cancer_path, contact_path = find_manual_mask_paths(input_folder, sample_name)

    for path, auto_mask, validation_type in [
        (car_t_path, car_t_labeled > 0, "CAR-T segmentation"),
        (cancer_path, cancer_labeled > 0, "Cancer segmentation"),
        (contact_path, combined_roi_mask, "Contact ROI"),
    ]:
        if path is None:
            continue
        metrics = compute_binary_validation_metrics(auto_mask, load_manual_binary_mask(path))
        metrics.update({
            "sample_name": sample_name,
            "validation_type": validation_type,
            "manual_mask_path": path,
        })
        rows.append(metrics)

    return rows


# =========================================================
# 7. VISUAL OUTPUTS
# =========================================================

def save_segmentation_debug(
    sample_name,
    channel_name,
    output_dir,
    raw_img,
    processed_img,
    binary_mask,
    labeled_img,
):
    """Save a 4-stage figure so segmentation quality can be checked visually."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title("Raw")
    axes[1].imshow(processed_img, cmap="gray")
    axes[1].set_title("Processed")
    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title("Binary mask")
    axes[3].imshow(labeled_img, cmap="nipy_spectral")
    axes[3].set_title(f"Final labels (n={int(labeled_img.max())})")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{sample_name} - {channel_name} segmentation debug")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{sample_name}_{channel_name}_segmentation_debug.png"),
        dpi=250,
    )
    plt.close(fig)


def save_seeded_segmentation_debug(
    sample_name,
    channel_label,
    file_token,
    output_dir,
    raw_img,
    processed_img,
    labeled_img,
    debug,
):
    """Save the seeded-watershed decision path, including markers."""
    fig, axes = plt.subplots(1, 7, figsize=(24, 4))

    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title("Raw")

    axes[1].imshow(processed_img, cmap="gray")
    axes[1].set_title("Background corrected")

    axes[2].imshow(debug["enhanced"], cmap="gray")
    axes[2].set_title("Mild CLAHE")

    axes[3].imshow(debug["strong_seeds"], cmap="gray")
    axes[3].set_title("Strong seed pixels")

    axes[4].imshow(debug["hysteresis"], cmap="gray")
    axes[4].set_title("Recovered territory")

    axes[5].imshow(debug["geometry"], cmap="gray")
    marker_coords = np.argwhere(debug["markers"] > 0)
    if marker_coords.size:
        axes[5].scatter(
            marker_coords[:, 1],
            marker_coords[:, 0],
            s=12,
            c="red",
            marker="x",
        )
    axes[5].set_title(f"Geometry + markers (n={int(debug['markers'].max())})")

    axes[6].imshow(labeled_img, cmap="nipy_spectral")
    axes[6].set_title(f"Final instances (n={int(labeled_img.max())})")

    for ax in axes:
        ax.axis("off")

    low_thr = debug.get("low_threshold", np.nan)
    high_thr = debug.get("high_threshold", np.nan)
    fig.suptitle(
        f"{sample_name} - {channel_label} seeded watershed "
        f"(low={low_thr:.3f}, high={high_thr:.3f})"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{sample_name}_{file_token}_segmentation_debug_v7.png"),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_overlay_image(
    sample_name,
    output_dir,
    bf_img,
    green_img,
    red_img,
    car_t_labeled,
    cancer_labeled,
    combined_roi_mask,
):
    green_display = normalize_image(green_img)
    red_display = normalize_image(red_img)

    if bf_img is not None:
        bf_display = normalize_image(bf_img)
        rgb = np.dstack([bf_display, bf_display, bf_display])
    else:
        rgb = np.zeros((green_img.shape[0], green_img.shape[1], 3), dtype=np.float32)
        rgb[..., 0] = red_display
        rgb[..., 1] = green_display

    car_t_boundary = segmentation.find_boundaries(car_t_labeled > 0, mode="outer")
    cancer_boundary = segmentation.find_boundaries(cancer_labeled > 0, mode="outer")

    rgb[car_t_boundary] = [0, 1, 0]
    rgb[cancer_boundary] = [1, 0, 0]
    rgb[combined_roi_mask] = [1, 1, 0]

    plt.figure(figsize=(7, 7))
    plt.imshow(np.clip(rgb, 0, 1))
    plt.title(f"{sample_name}: segmentation and contact ROI")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sample_name}_overlay.png"), dpi=300)
    plt.close()


def save_segmentation_masks(sample_name, output_dir, car_t_labeled, cancer_labeled, combined_roi_mask):
    plt.figure(figsize=(7, 7))
    plt.imshow(car_t_labeled, cmap="nipy_spectral")
    plt.title(f"{sample_name}: CAR-T segmentation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sample_name}_car_t_segmentation.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(cancer_labeled, cmap="nipy_spectral")
    plt.title(f"{sample_name}: Cancer segmentation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sample_name}_cancer_segmentation.png"), dpi=300)
    plt.close()

    imsave(
        os.path.join(output_dir, f"{sample_name}_contact_roi_mask.png"),
        combined_roi_mask.astype(np.uint8) * 255,
    )


# =========================================================
# 8. ANALYZE ONE SAMPLE
# =========================================================

def analyze_one_sample(
    group,
    output_dir,
    pixel_size_um=0.108,

    # CAR-T v7: recover faint borders but keep adjacent cells separated
    min_size_car_t=80,
    max_size_car_t=None,
    threshold_method_car_t="otsu",
    smooth_sigma_car_t=1.0,
    background_sigma_car_t=35.0,
    clahe_clip_limit_car_t=0.010,
    clahe_kernel_size_car_t=None,
    low_threshold_scale_car_t=0.55,
    high_threshold_scale_car_t=0.95,
    closing_radius_car_t=1,
    geometry_hole_area_car_t=1200,
    distance_smooth_sigma_car_t=1.5,
    marker_min_distance_car_t=12,
    secondary_marker_min_distance_car_t=7,
    merged_area_factor_car_t=1.8,
    resplit_area_factor_car_t=2.5,
    resplit_min_peak_distance_car_t=8,
    resplit_min_daughter_fraction_car_t=0.30,

    # Cancer v7: the same seeded-watershed strategy as CAR-T,
    # but with its own channel-specific parameters.
    min_size_cancer=250,
    max_size_cancer=None,
    threshold_method_cancer="otsu",
    smooth_sigma_cancer=1.0,
    background_sigma_cancer=50.0,
    clahe_clip_limit_cancer=0.010,
    clahe_kernel_size_cancer=None,
    low_threshold_scale_cancer=0.55,
    high_threshold_scale_cancer=0.95,
    closing_radius_cancer=1,
    geometry_hole_area_cancer=1200,
    distance_smooth_sigma_cancer=1.5,
    marker_min_distance_cancer=12,
    secondary_marker_min_distance_cancer=7,
    merged_area_factor_cancer=1.8,
    resplit_area_factor_cancer=3.0,
    resplit_min_peak_distance_cancer=8,
    resplit_min_daughter_fraction_cancer=0.30,

    min_solidity_car_t=0.0,
    min_solidity_cancer=0.0,
    max_eccentricity_car_t=1.0,
    max_eccentricity_cancer=1.0,

    contact_distance_px=2,
    possible_distance_px=8,
    min_contact_overlap_px=25,
    min_contact_overlap_fraction=0.02,
    roi_dilation_px=2,
    maximum_search_distance_px=40,
):
    sample_name = group["sample_name"]
    input_folder = os.path.dirname(group["green_path"])
    print(f"\nAnalyzing sample: {sample_name}")

    bf_img = load_grayscale_image(group["bf_path"]) if group["bf_path"] else None
    green_img = load_grayscale_image(group["green_path"])
    red_img = load_grayscale_image(group["red_path"])

    if green_img.shape != red_img.shape:
        raise ValueError(f"Green and red images must have same shape: {green_img.shape} vs {red_img.shape}")
    if bf_img is not None and bf_img.shape != green_img.shape:
        raise ValueError("Bright-field image must have same shape as fluorescence images")

    image_shape = green_img.shape

    car_t_labeled, car_t_processed, car_t_binary, car_t_debug = segment_seeded_instances(
        green_img,
        min_size=min_size_car_t,
        max_size=max_size_car_t,
        threshold_method=threshold_method_car_t,
        smooth_sigma=smooth_sigma_car_t,
        background_sigma=background_sigma_car_t,
        clahe_clip_limit=clahe_clip_limit_car_t,
        clahe_kernel_size=clahe_kernel_size_car_t,
        low_threshold_scale=low_threshold_scale_car_t,
        high_threshold_scale=high_threshold_scale_car_t,
        closing_radius=closing_radius_car_t,
        geometry_hole_area=geometry_hole_area_car_t,
        distance_smooth_sigma=distance_smooth_sigma_car_t,
        marker_min_distance=marker_min_distance_car_t,
        secondary_marker_min_distance=secondary_marker_min_distance_car_t,
        merged_area_factor=merged_area_factor_car_t,
        enable_large_instance_resplit=True,
        resplit_area_factor=resplit_area_factor_car_t,
        resplit_min_peak_distance=resplit_min_peak_distance_car_t,
        resplit_min_daughter_fraction=resplit_min_daughter_fraction_car_t,
        min_solidity=min_solidity_car_t,
        max_eccentricity=max_eccentricity_car_t,
    )

    cancer_labeled, cancer_processed, cancer_binary, cancer_debug = segment_seeded_instances(
        red_img,
        min_size=min_size_cancer,
        max_size=max_size_cancer,
        threshold_method=threshold_method_cancer,
        smooth_sigma=smooth_sigma_cancer,
        background_sigma=background_sigma_cancer,
        clahe_clip_limit=clahe_clip_limit_cancer,
        clahe_kernel_size=clahe_kernel_size_cancer,
        low_threshold_scale=low_threshold_scale_cancer,
        high_threshold_scale=high_threshold_scale_cancer,
        closing_radius=closing_radius_cancer,
        geometry_hole_area=geometry_hole_area_cancer,
        distance_smooth_sigma=distance_smooth_sigma_cancer,
        marker_min_distance=marker_min_distance_cancer,
        secondary_marker_min_distance=secondary_marker_min_distance_cancer,
        merged_area_factor=merged_area_factor_cancer,
        enable_large_instance_resplit=True,
        resplit_area_factor=resplit_area_factor_cancer,
        resplit_min_peak_distance=resplit_min_peak_distance_cancer,
        resplit_min_daughter_fraction=resplit_min_daughter_fraction_cancer,
        min_solidity=min_solidity_cancer,
        max_eccentricity=max_eccentricity_cancer,
    )

    save_seeded_segmentation_debug(
        sample_name=sample_name,
        channel_label="CAR-T",
        file_token="car_t",
        output_dir=output_dir,
        raw_img=green_img,
        processed_img=car_t_processed,
        labeled_img=car_t_labeled,
        debug=car_t_debug,
    )
    save_seeded_segmentation_debug(
        sample_name=sample_name,
        channel_label="Cancer",
        file_token="cancer",
        output_dir=output_dir,
        raw_img=red_img,
        processed_img=cancer_processed,
        labeled_img=cancer_labeled,
        debug=cancer_debug,
    )

    car_t_cells = extract_cells(
        car_t_labeled, green_img, "CAR-T", pixel_size_um=pixel_size_um
    )
    cancer_cells = extract_cells(
        cancer_labeled, red_img, "cancer", pixel_size_um=pixel_size_um
    )

    # v7: keep all detections, but flag suspicious/border objects in the CSV.
    car_t_cells = annotate_cell_quality(
        car_t_cells,
        small_area_factor=0.35,
        large_area_factor=resplit_area_factor_car_t,
        image_shape=image_shape,
    )
    cancer_cells = annotate_cell_quality(
        cancer_cells,
        small_area_factor=0.35,
        large_area_factor=resplit_area_factor_cancer,
        image_shape=image_shape,
    )

    # Print a direct sanity check for this version.
    if car_t_cells:
        med = np.median([c["area_um2"] for c in car_t_cells])
        print(f"CAR-T: n={len(car_t_cells)}, median segmented area={med:.3f} um^2")
    else:
        print("CAR-T: n=0")

    if cancer_cells:
        med = np.median([c["area_um2"] for c in cancer_cells])
        print(f"Cancer: n={len(cancer_cells)}, median segmented area={med:.3f} um^2")
    else:
        print("Cancer: n=0")

    cell_rows = []
    for cell in car_t_cells + cancer_cells:
        cell_rows.append({
            "sample_name": sample_name,
            "cell_type": cell["cell_type"],
            "cell_id": cell["cell_id"],
            "area_px": cell["area_px"],
            "area_um2": cell["area_um2"],
            "centroid_y": cell["centroid_y"],
            "centroid_x": cell["centroid_x"],
            "mean_intensity": cell["mean_intensity"],
            "total_intensity": cell["total_intensity"],
            "integrated_density": cell["integrated_density"],
            "local_background_mean": cell["local_background_mean"],
            "corrected_mean_intensity": cell["corrected_mean_intensity"],
            "corrected_total_intensity": cell["corrected_total_intensity"],
            "corrected_integrated_density": cell["corrected_integrated_density"],
            "touches_border": cell.get("touches_border", False),
            "area_ratio_to_median": cell.get("area_ratio_to_median", np.nan),
            "cell_quality": cell.get("cell_quality", "unknown"),
        })

    interaction_rows = []
    roi_masks_for_overlay = []

    for car_t in car_t_cells:
        for cancer in cancer_cells:
            centroid_distance_px = np.hypot(
                car_t["centroid_x"] - cancer["centroid_x"],
                car_t["centroid_y"] - cancer["centroid_y"],
            )

            # v7 SAFE PREFILTER:
            # Do NOT reject a pair using a fixed centroid-distance threshold.
            # Large cells can physically touch even when their centroids are
            # >40 px apart (this happened in sample0). Use a cell-size-aware
            # geometric pre-filter instead.
            pair_is_possible, adaptive_search_limit_px = is_geometrically_possible_pair(
                car_t_cell=car_t,
                cancer_cell=cancer,
                centroid_distance_px=centroid_distance_px,
                possible_distance_px=possible_distance_px,
                safety_margin_px=2,
            )

            if not pair_is_possible:
                continue

            result = classify_contact(
                car_t_cell=car_t,
                cancer_cell=cancer,
                image_shape=image_shape,
                response_image_red=red_img,
                response_image_green=green_img,
                contact_distance_px=contact_distance_px,
                possible_distance_px=possible_distance_px,
                roi_dilation_px=roi_dilation_px,
                pixel_size_um=pixel_size_um,
                min_contact_overlap_px=min_contact_overlap_px,
                min_contact_overlap_fraction=min_contact_overlap_fraction,
            )

            # Yellow ROI in the overlay is reserved for CONFIRMED contact.
            # Possible contacts remain in the CSV/summary graph but are not
            # painted as if they were definite interactions.
            if result["contact_happened"] and result["roi_area_px"] > 0:
                roi_masks_for_overlay.append(result["roi_mask"])

            interaction_rows.append({
                "sample_name": sample_name,
                "car_t_cell_id": car_t["cell_id"],
                "cancer_cell_id": cancer["cell_id"],
                "centroid_distance_px": float(centroid_distance_px),
                "centroid_distance_um": float(centroid_distance_px * pixel_size_um),
                "adaptive_search_limit_px": float(adaptive_search_limit_px),
                "contact_status": result["contact_status"],
                "contact_reason": result["contact_reason"],
                "contact_happened": result["contact_happened"],
                "possible_contact": result["possible_contact"],
                "min_distance_px": result["min_distance_px"],
                "min_distance_um": result["min_distance_um"],
                "overlap_area_px": result["overlap_area_px"],
                "overlap_area_um2": result["overlap_area_um2"],
                "overlap_fraction_smaller_cell": result["overlap_fraction_smaller_cell"],
                "required_overlap_px": result["required_overlap_px"],
                "roi_area_px": result["roi_area_px"],
                "roi_area_um2": result["roi_area_um2"],
                "roi_mean_intensity": result["roi_mean_intensity"],
                "roi_total_intensity": result["roi_total_intensity"],
                "roi_integrated_density": result["roi_integrated_density"],
                "roi_max_intensity": result["roi_max_intensity"],
                "roi_background_mean": result["roi_background_mean"],
                "roi_corrected_mean_intensity": result["roi_corrected_mean_intensity"],
                "roi_corrected_total_intensity": result["roi_corrected_total_intensity"],
                "roi_corrected_integrated_density": result["roi_corrected_integrated_density"],
                "car_t_area_px": car_t["area_px"],
                "car_t_area_um2": car_t["area_um2"],
                "car_t_mean_intensity": car_t["mean_intensity"],
                "car_t_total_intensity": car_t["total_intensity"],
                "car_t_integrated_density": car_t["integrated_density"],
                "car_t_local_background_mean": car_t["local_background_mean"],
                "car_t_corrected_mean_intensity": car_t["corrected_mean_intensity"],
                "car_t_corrected_total_intensity": car_t["corrected_total_intensity"],
                "car_t_corrected_integrated_density": car_t["corrected_integrated_density"],
                "car_t_cell_quality": car_t.get("cell_quality", "unknown"),
                "car_t_touches_border": car_t.get("touches_border", False),
                "cancer_area_px": cancer["area_px"],
                "cancer_area_um2": cancer["area_um2"],
                "cancer_mean_intensity": cancer["mean_intensity"],
                "cancer_total_intensity": cancer["total_intensity"],
                "cancer_integrated_density": cancer["integrated_density"],
                "cancer_local_background_mean": cancer["local_background_mean"],
                "cancer_corrected_mean_intensity": cancer["corrected_mean_intensity"],
                "cancer_corrected_total_intensity": cancer["corrected_total_intensity"],
                "cancer_corrected_integrated_density": cancer["corrected_integrated_density"],
                "cancer_cell_quality": cancer.get("cell_quality", "unknown"),
                "cancer_touches_border": cancer.get("touches_border", False),
            })

    combined_roi_mask = np.zeros(image_shape, dtype=bool)
    for roi_mask in roi_masks_for_overlay:
        combined_roi_mask |= roi_mask

    validation_rows = validate_sample(
        sample_name,
        input_folder,
        car_t_labeled,
        cancer_labeled,
        combined_roi_mask,
    )

    save_overlay_image(
        sample_name,
        output_dir,
        bf_img,
        green_img,
        red_img,
        car_t_labeled,
        cancer_labeled,
        combined_roi_mask,
    )
    save_segmentation_masks(
        sample_name,
        output_dir,
        car_t_labeled,
        cancer_labeled,
        combined_roi_mask,
    )

    print(f"Nearby cell pairs analyzed: {len(interaction_rows)}")
    return cell_rows, interaction_rows, validation_rows


# =========================================================
# 9. SUMMARY GRAPHS
# =========================================================

def save_contacts_summary_graph(interactions_df, output_dir):
    if interactions_df is None or len(interactions_df) == 0:
        return

    summary = (
        interactions_df.groupby(["sample_name", "contact_status"])
        .size().reset_index(name="count")
    )
    pivot = summary.pivot(index="sample_name", columns="contact_status", values="count").fillna(0)

    for col in ["contact_happened", "possible_contact", "no_contact"]:
        if col not in pivot.columns:
            pivot[col] = 0

    plot_df = pivot[["contact_happened", "possible_contact"]]
    x = np.arange(len(plot_df.index))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, plot_df["contact_happened"], width, label="Contact happened")
    plt.bar(x + width / 2, plot_df["possible_contact"], width, label="Possible contact")
    plt.xticks(x, plot_df.index, rotation=45, ha="right")
    plt.ylabel("Number of interactions")
    plt.xlabel("Sample")
    plt.title("Detected interactions per sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contacts_summary_graph.png"), dpi=300)
    plt.close()


def save_validation_summary_graph(validation_df, output_dir):
    if validation_df is None or len(validation_df) == 0:
        return

    metrics = [m for m in ["precision", "recall", "iou", "dice"] if m in validation_df.columns]
    if not metrics:
        return

    plot_df = validation_df.copy()
    plot_df["label"] = plot_df["sample_name"] + " - " + plot_df["validation_type"]
    x = np.arange(len(plot_df))
    width = 0.18

    plt.figure(figsize=(11, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - len(metrics) / 2) * width, plot_df[metric], width, label=metric)
    plt.xticks(x, plot_df["label"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Validation metrics vs manual masks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_summary_graph.png"), dpi=300)
    plt.close()



def save_overall_validation_graph(overall_validation_df, output_dir):
    """Save a line-plot summary of aggregate validation across all samples."""
    if overall_validation_df is None or len(overall_validation_df) == 0:
        return

    plot_df = overall_validation_df[
        overall_validation_df["validation_type"] != "ALL VALIDATION MASKS"
    ].copy()

    if len(plot_df) == 0:
        return

    metrics = [
        ("macro_precision", "Precision"),
        ("macro_recall", "Recall"),
        ("macro_iou", "IoU"),
        ("macro_dice", "Dice"),
    ]

    x = np.arange(len(plot_df))
    plt.figure(figsize=(9, 5))

    for col, label in metrics:
        if col in plot_df.columns:
            plt.plot(x, plot_df[col], marker="o", linewidth=1.8, label=label)

    plt.xticks(x, plot_df["validation_type"], rotation=25, ha="right")
    plt.ylabel("Mean validation score across samples")
    plt.xlabel("Validation task")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "overall_validation_across_samples.png"),
        dpi=300,
    )
    plt.close()


# =========================================================
# 10. FULL PIPELINE
# =========================================================

def analyze_grayscale_image_folder(
    input_folder,
    output_dir="results_grayscale_images_v7_conservative_contact_resplit",
    pixel_size_um=0.108,

    min_size_car_t=80,
    min_size_cancer=150,
    max_size_car_t=None,
    max_size_cancer=None,

    threshold_method_car_t="otsu",
    threshold_method_cancer="otsu",

    smooth_sigma_car_t=1.0,
    smooth_sigma_cancer=1.0,
    background_sigma_car_t=35.0,
    background_sigma_cancer=50.0,

    # CAR-T v5 seeded-watershed parameters
    clahe_clip_limit_car_t=0.010,
    clahe_kernel_size_car_t=None,
    low_threshold_scale_car_t=0.55,
    high_threshold_scale_car_t=0.95,
    closing_radius_car_t=1,
    geometry_hole_area_car_t=1200,
    distance_smooth_sigma_car_t=1.5,
    marker_min_distance_car_t=12,
    secondary_marker_min_distance_car_t=7,
    merged_area_factor_car_t=1.8,
    resplit_area_factor_car_t=2.5,
    resplit_min_peak_distance_car_t=8,
    resplit_min_daughter_fraction_car_t=0.30,

    # Cancer: same seeded-watershed strategy, tuned independently
    clahe_clip_limit_cancer=0.010,
    clahe_kernel_size_cancer=None,
    low_threshold_scale_cancer=0.55,
    high_threshold_scale_cancer=0.95,
    closing_radius_cancer=1,
    geometry_hole_area_cancer=1200,
    distance_smooth_sigma_cancer=1.5,
    marker_min_distance_cancer=12,
    secondary_marker_min_distance_cancer=7,
    merged_area_factor_cancer=1.8,
    resplit_area_factor_cancer=3.0,
    resplit_min_peak_distance_cancer=8,
    resplit_min_daughter_fraction_cancer=0.30,

    min_solidity_car_t=0.0,
    min_solidity_cancer=0.0,
    max_eccentricity_car_t=1.0,
    max_eccentricity_cancer=1.0,

    contact_distance_px=2,
    possible_distance_px=8,
    min_contact_overlap_px=25,
    min_contact_overlap_fraction=0.02,
    roi_dilation_px=2,
    maximum_search_distance_px=40,
):
    os.makedirs(output_dir, exist_ok=True)
    groups = organize_grayscale_images(input_folder)

    all_cell_rows = []
    all_interaction_rows = []
    all_validation_rows = []

    for group in groups:
        cell_rows, interaction_rows, validation_rows = analyze_one_sample(
            group=group,
            output_dir=output_dir,
            pixel_size_um=pixel_size_um,

            min_size_car_t=min_size_car_t,
            min_size_cancer=min_size_cancer,
            max_size_car_t=max_size_car_t,
            max_size_cancer=max_size_cancer,

            threshold_method_car_t=threshold_method_car_t,
            threshold_method_cancer=threshold_method_cancer,

            smooth_sigma_car_t=smooth_sigma_car_t,
            smooth_sigma_cancer=smooth_sigma_cancer,
            background_sigma_car_t=background_sigma_car_t,
            background_sigma_cancer=background_sigma_cancer,

            clahe_clip_limit_car_t=clahe_clip_limit_car_t,
            clahe_kernel_size_car_t=clahe_kernel_size_car_t,
            low_threshold_scale_car_t=low_threshold_scale_car_t,
            high_threshold_scale_car_t=high_threshold_scale_car_t,
            closing_radius_car_t=closing_radius_car_t,
            geometry_hole_area_car_t=geometry_hole_area_car_t,
            distance_smooth_sigma_car_t=distance_smooth_sigma_car_t,
            marker_min_distance_car_t=marker_min_distance_car_t,
            secondary_marker_min_distance_car_t=secondary_marker_min_distance_car_t,
            merged_area_factor_car_t=merged_area_factor_car_t,
            resplit_area_factor_car_t=resplit_area_factor_car_t,
            resplit_min_peak_distance_car_t=resplit_min_peak_distance_car_t,
            resplit_min_daughter_fraction_car_t=resplit_min_daughter_fraction_car_t,

            clahe_clip_limit_cancer=clahe_clip_limit_cancer,
            clahe_kernel_size_cancer=clahe_kernel_size_cancer,
            low_threshold_scale_cancer=low_threshold_scale_cancer,
            high_threshold_scale_cancer=high_threshold_scale_cancer,
            closing_radius_cancer=closing_radius_cancer,
            geometry_hole_area_cancer=geometry_hole_area_cancer,
            distance_smooth_sigma_cancer=distance_smooth_sigma_cancer,
            marker_min_distance_cancer=marker_min_distance_cancer,
            secondary_marker_min_distance_cancer=secondary_marker_min_distance_cancer,
            merged_area_factor_cancer=merged_area_factor_cancer,
            resplit_area_factor_cancer=resplit_area_factor_cancer,
            resplit_min_peak_distance_cancer=resplit_min_peak_distance_cancer,
            resplit_min_daughter_fraction_cancer=resplit_min_daughter_fraction_cancer,

            min_solidity_car_t=min_solidity_car_t,
            min_solidity_cancer=min_solidity_cancer,
            max_eccentricity_car_t=max_eccentricity_car_t,
            max_eccentricity_cancer=max_eccentricity_cancer,

            contact_distance_px=contact_distance_px,
            possible_distance_px=possible_distance_px,
            min_contact_overlap_px=min_contact_overlap_px,
            min_contact_overlap_fraction=min_contact_overlap_fraction,
            roi_dilation_px=roi_dilation_px,
            maximum_search_distance_px=maximum_search_distance_px,
        )

        all_cell_rows.extend(cell_rows)
        all_interaction_rows.extend(interaction_rows)
        all_validation_rows.extend(validation_rows)

    cells_df = pd.DataFrame(all_cell_rows)
    interactions_df = pd.DataFrame(all_interaction_rows)
    validation_df = pd.DataFrame(all_validation_rows)
    overall_validation_df = compute_overall_validation_summary(validation_df)

    timestamp = int(time.time())
    cells_path = os.path.join(output_dir, f"cell_measurements_{timestamp}.csv")
    interactions_path = os.path.join(output_dir, f"interaction_measurements_{timestamp}.csv")
    validation_path = os.path.join(output_dir, f"validation_metrics_{timestamp}.csv")
    overall_validation_path = os.path.join(
        output_dir,
        f"validation_overall_metrics_{timestamp}.csv"
    )

    cells_df.to_csv(cells_path, index=False)
    interactions_df.to_csv(interactions_path, index=False)
    if len(validation_df) > 0:
        validation_df.to_csv(validation_path, index=False)
        overall_validation_df.to_csv(overall_validation_path, index=False)

    save_contacts_summary_graph(interactions_df, output_dir)
    save_validation_summary_graph(validation_df, output_dir)
    save_overall_validation_graph(overall_validation_df, output_dir)

    print("\nAnalysis complete.")
    print("Cell measurements:", cells_path)
    print("Interaction measurements:", interactions_path)
    if len(validation_df) > 0:
        print("Validation metrics:", validation_path)
        print("Overall validation metrics:", overall_validation_path)

        grand = overall_validation_df[
            overall_validation_df["validation_type"] == "ALL VALIDATION MASKS"
        ]
        if len(grand) > 0:
            grand = grand.iloc[0]
            print("\nOVERALL VALIDATION ACROSS ALL AVAILABLE MANUAL MASKS")
            print(f"Validated samples: {int(grand['n_samples'])}")
            print(f"Validated masks: {int(grand['n_validated_masks'])}")
            print(f"Overall success score (mean Dice): {grand['success_score_percent']:.2f}%")
            print(f"Mean Precision: {grand['macro_precision']:.4f}")
            print(f"Mean Recall: {grand['macro_recall']:.4f}")
            print(f"Mean IoU: {grand['macro_iou']:.4f}")
            print(f"Mean Dice: {grand['macro_dice']:.4f}")

        print("\nVALIDATION BY TASK ACROSS ALL SAMPLES")
        for _, row in overall_validation_df.iterrows():
            if row["validation_type"] == "ALL VALIDATION MASKS":
                continue
            print(
                f"{row['validation_type']}: "
                f"Success/Dice={row['success_score_percent']:.2f}% | "
                f"Precision={row['macro_precision']:.3f} | "
                f"Recall={row['macro_recall']:.3f} | "
                f"IoU={row['macro_iou']:.3f}"
            )
    else:
        print("No manual masks found; validation CSV was not created.")
    print("Debug images, masks, overlays and graphs saved in:", output_dir)

    return cells_df, interactions_df, validation_df, overall_validation_df


# =========================================================
# 11. RUN
# =========================================================

if __name__ == "__main__":
    input_folder = "data/images_folder"
    output_dir = "results_grayscale_images_v11_combined_roi_only"

    print("Input folder:", input_folder)
    print("Exists?", os.path.exists(input_folder))

    cells_df, interactions_df, validation_df, overall_validation_df = analyze_grayscale_image_folder(
        input_folder=input_folder,
        output_dir=output_dir,
        pixel_size_um=0.108,

        # CAR-T v7: faint-border recovery + strict instance separation
        min_size_car_t=80,
        background_sigma_car_t=35.0,
        clahe_clip_limit_car_t=0.010,
        low_threshold_scale_car_t=0.55,
        high_threshold_scale_car_t=0.95,
        closing_radius_car_t=1,
        geometry_hole_area_car_t=1200,
        distance_smooth_sigma_car_t=1.5,
        marker_min_distance_car_t=12,
        secondary_marker_min_distance_car_t=7,
        merged_area_factor_car_t=1.8,
        resplit_area_factor_car_t=2.5,
        resplit_min_peak_distance_car_t=8,
        resplit_min_daughter_fraction_car_t=0.30,

        # Cancer: same faint-border recovery + instance separation
        # as CAR-T, but with cancer-specific filtering.
        min_size_cancer=250,
        background_sigma_cancer=50.0,
        clahe_clip_limit_cancer=0.010,
        low_threshold_scale_cancer=0.55,
        high_threshold_scale_cancer=0.95,
        closing_radius_cancer=1,
        geometry_hole_area_cancer=1200,
        distance_smooth_sigma_cancer=1.5,
        marker_min_distance_cancer=12,
        secondary_marker_min_distance_cancer=7,
        merged_area_factor_cancer=1.8,
        resplit_area_factor_cancer=3.0,
        resplit_min_peak_distance_cancer=8,
        resplit_min_daughter_fraction_cancer=0.30,

        # v9 conservative contact logic + dual-channel ROI fluorescence:
        # definite contact requires meaningful mask overlap.
        contact_distance_px=2,
        possible_distance_px=8,
        min_contact_overlap_px=25,
        min_contact_overlap_fraction=0.02,
        roi_dilation_px=2,
        maximum_search_distance_px=40,
    )

    print("\nCell measurements preview:")
    print(cells_df.head())

    print("\nInteraction measurements preview:")
    print(interactions_df.head())

    print("\nValidation measurements preview:")
    print(validation_df.head())

    print("\nOverall validation preview:")
    print(overall_validation_df)
