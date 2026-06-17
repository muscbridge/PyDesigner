#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import binary_erosion, generate_binary_structure


# ============================================================
# USER INPUTS
# ============================================================

base_dir = "/Volumes/Flashy/HIE_FBI_003/FBWM_b4000"

metrics_dir = os.path.join(base_dir, "metrics")
jhu_dir = os.path.join("/Volumes/Flashy/HIE_FBI_003", "jhu")
png_dir = os.path.join("/Volumes/Flashy/HIE_FBI_003", "png")

os.makedirs(png_dir, exist_ok=True)

# Base image
b0_path = os.path.join(base_dir, "B0.nii")

# Combined WM mask
# wm_mask_path = os.path.join(base_dir, "wm_masked.nii")
wm_mask_path = os.path.join(base_dir, "brain_mask_eroF_noCSF.nii")

# ROI masks
roi_paths = {
    "gcc":   os.path.join(jhu_dir, "jhu_to_fa_gcc_mask.nii"),
    "bcc":   os.path.join(jhu_dir, "jhu_to_fa_bcc_mask.nii"),
    "scc":   os.path.join(jhu_dir, "jhu_to_fa_scc_mask.nii"),
    "plicl": os.path.join(jhu_dir, "jhu_to_fa_plicl_mask.nii"),
    "plicr": os.path.join(jhu_dir, "jhu_to_fa_plicr_mask.nii"),
}

# Metric maps
metric_paths = {
    "MD":  os.path.join(metrics_dir, "dti_md.nii"),
    "FA":  os.path.join(metrics_dir, "dti_fa.nii"),
    "MK":  os.path.join(metrics_dir, "dki_mk.nii"),
    "KFA": os.path.join(metrics_dir, "dki_kfa.nii"),
    "FAA": os.path.join(metrics_dir, "fbi_faa.nii"),
}

# Output file
panel_out = os.path.join(png_dir, "metric_roi_panel_2x3.png")

# Fixed display ranges for metric maps
metric_ranges = {
    "MD":  (0, 3.0),
    "FA":  (0, 1.0),
    "MK":  (0, 1.0),
    "KFA": (0, 1.0),
    "FAA": (0, 1.0),
}

# Metric colormaps
metric_cmaps = {
    "MD":  "magma",
    "FA":  "magma",
    "MK":  "magma",
    "KFA": "magma",
    "FAA": "magma",
}

# ROI colors
roi_colors = {
    "gcc":   "lime",
    # "bcc":   "blue",
    "scc":   "red",
    "plicl": "blue",
    "plicr": "blue",
}

# ROI erosion settings
# Set roi_erosion_iterations = 0 to disable erosion.
roi_erosion_iterations = 1

# Connectivity for 3D erosion:
# 1 = face-connected, conservative erosion
# 2 = edge-connected
# 3 = corner-connected, more aggressive erosion
roi_erosion_connectivity = 1

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_nifti(path):
    """
    Load a NIfTI image and return data array and image object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    img = nib.load(path)
    data = img.get_fdata()
    return data, img


def ensure_3d(data):
    """
    If image is 4D, average across the last axis.
    Otherwise return unchanged.
    """
    if data.ndim == 4:
        return np.mean(data, axis=-1)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D image, got shape {data.shape}")

    return data


def binarize_mask(mask):
    """
    Convert any mask-like image into a boolean mask.
    """
    return np.isfinite(mask) & (mask > 0)


def choose_best_slice(mask3d, axis=2):
    """
    Choose the slice index with the largest number of mask voxels
    along a given axis.

    axis=0 sagittal
    axis=1 coronal
    axis=2 axial
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    sum_axes = tuple(i for i in range(3) if i != axis)
    counts = np.sum(mask3d, axis=sum_axes)

    if np.max(counts) == 0:
        raise ValueError("Mask is empty; cannot choose best slice.")

    return int(np.argmax(counts))


def get_slice(data, axis, idx, rotate=True):
    """
    Extract a 2D slice.

    axis=0 sagittal
    axis=1 coronal
    axis=2 axial

    rotate=True applies np.rot90 for display.
    """
    if axis == 0:
        sl = data[idx, :, :]
    elif axis == 1:
        sl = data[:, idx, :]
    elif axis == 2:
        sl = data[:, :, idx]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    if rotate:
        sl = np.rot90(sl)

    return sl


def robust_limits(data, mask=None, lower=1, upper=99):
    """
    Compute robust display limits using percentiles.
    """
    if mask is not None:
        vals = data[np.isfinite(data) & mask]
    else:
        vals = data[np.isfinite(data)]

    if vals.size == 0:
        raise ValueError("No finite values found for robust limits.")

    vmin, vmax = np.percentile(vals, [lower, upper])

    if vmin == vmax:
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)

    return float(vmin), float(vmax)


def make_rgba_overlay(mask2d, color, alpha=0.45):
    """
    Create an RGBA overlay image from a 2D boolean mask.
    """
    rgba = np.zeros(mask2d.shape + (4,), dtype=float)
    rgba[..., :] = 0

    c = to_rgba(color, alpha=alpha)
    rgba[mask2d, :] = c

    return rgba


def masked_metric(metric, wm_mask):
    """
    Restrict metric map to white matter mask.
    Values outside the mask are set to NaN.
    """
    out = np.array(metric, copy=True)
    out[~wm_mask] = np.nan
    return out


def check_same_shape(reference, data, name):
    """
    Ensure all input volumes have matching spatial dimensions.
    """
    if reference.shape != data.shape:
        raise ValueError(
            f"Shape mismatch for {name}: expected {reference.shape}, got {data.shape}"
        )

def crop_bounds_from_mask(mask2d, pad=10):
    """
    Compute a bounding box around nonzero/True pixels in a 2D mask.

    Returns:
        row_min, row_max, col_min, col_max

    pad is added on all sides, clipped to image bounds.
    """
    rows, cols = np.where(mask2d)

    if rows.size == 0 or cols.size == 0:
        # If mask is empty, return full image
        return 0, mask2d.shape[0], 0, mask2d.shape[1]

    rmin = max(int(rows.min()) - pad, 0)
    rmax = min(int(rows.max()) + pad + 1, mask2d.shape[0])
    cmin = max(int(cols.min()) - pad, 0)
    cmax = min(int(cols.max()) + pad + 1, mask2d.shape[1])

    return rmin, rmax, cmin, cmax


def apply_crop(img2d, crop_bounds):
    """
    Apply a 2D crop to an image or mask.
    """
    rmin, rmax, cmin, cmax = crop_bounds
    return img2d[rmin:rmax, cmin:cmax]

def erode_roi_mask(mask3d, iterations=1, connectivity=1):
    """
    Erode a 3D boolean ROI mask.

    Parameters
    ----------
    mask3d : np.ndarray
        Boolean 3D ROI mask.
    iterations : int
        Number of erosion iterations.
        Use 0 to disable erosion.
    connectivity : int
        3D structuring element connectivity.
        1 = face-connected, conservative
        2 = edge-connected
        3 = corner-connected, more aggressive

    Returns
    -------
    eroded : np.ndarray
        Eroded boolean ROI mask.
    """
    if iterations <= 0:
        return mask3d

    structure = generate_binary_structure(rank=3, connectivity=connectivity)

    eroded = binary_erosion(
        mask3d,
        structure=structure,
        iterations=iterations,
        border_value=0
    )

    return eroded

# ============================================================
# LOAD DATA
# ============================================================

print("Loading B0...")
b0, _ = load_nifti(b0_path)
b0 = ensure_3d(b0)

print("Loading WM mask...")
wm_mask, _ = load_nifti(wm_mask_path)
wm_mask = ensure_3d(wm_mask)
wm_mask = binarize_mask(wm_mask)

check_same_shape(b0, wm_mask, "WM mask")

# Load, erode, and WM-mask ROIs
print("Loading ROI masks...")
roi_data = {}

for name, path in roi_paths.items():
    arr, _ = load_nifti(path)
    arr = ensure_3d(arr)
    check_same_shape(b0, arr, f"ROI {name}")

    roi_mask_orig = binarize_mask(arr)

    roi_mask_eroded = erode_roi_mask(
        roi_mask_orig,
        iterations=roi_erosion_iterations,
        connectivity=roi_erosion_connectivity
    )

    roi_mask_eroded = roi_mask_eroded & wm_mask

    # Fallback if erosion removes the ROI entirely
    if np.sum(roi_mask_eroded) == 0:
        print(
            f"Warning: erosion removed ROI {name}; "
            "using original ROI intersected with WM mask instead."
        )
        roi_data[name] = roi_mask_orig & wm_mask
    else:
        roi_data[name] = roi_mask_eroded

    print(
        f"ROI {name}: "
        f"{np.sum(roi_mask_orig)} voxels before erosion, "
        f"{np.sum(roi_data[name])} voxels after erosion + WM mask"
    )

# Union ROI mask for slice selection
roi_union = np.zeros_like(wm_mask, dtype=bool)

for arr in roi_data.values():
    roi_union |= arr

# Choose axial display slice based on ROI burden
axi_idx = choose_best_slice(roi_union, axis=2)

print(f"Axial slice selected from ROI union: {axi_idx}")

# Load metric maps
print("Loading metric maps...")
metric_data = {}

for name, path in metric_paths.items():
    arr, _ = load_nifti(path)
    arr = ensure_3d(arr)
    check_same_shape(b0, arr, f"Metric {name}")

    arr = masked_metric(arr, wm_mask)
    metric_data[name] = arr


# ============================================================
# CREATE 2 x 3 PANEL FIGURE WITH BLACK BACKGROUND
# ============================================================

print("Creating 2 x 3 panel figure...")

panel_order = [
    ("MD", "metric"),
    ("FA", "metric"),
    ("MK", "metric"),
    ("KFA", "metric"),
    ("ROI overlay", "roi"),
    ("FAA", "metric"),
]

fig, axes = plt.subplots(
    3,
    2,
    figsize=(8, 10),
    constrained_layout=True,
    facecolor="black"
)

axes = axes.ravel()

# B0 display limits only needed for ROI panel
b0_vmin, b0_vmax = robust_limits(b0, wm_mask)

# Build a 2D crop mask from the axial WM mask and ROI union.
# This ensures all panels are cropped identically.
wm_axi = get_slice(wm_mask, axis=2, idx=axi_idx)
roi_union_axi = get_slice(roi_union, axis=2, idx=axi_idx)

crop_mask = wm_axi | roi_union_axi

# Increase pad for more surrounding anatomy, decrease for tighter crop.
crop_pad = 4
crop_bounds = crop_bounds_from_mask(crop_mask, pad=crop_pad)

for ax, (title, panel_type) in zip(axes, panel_order):

    # Make each subplot background black
    ax.set_facecolor("black")

    if panel_type == "metric":
        metric_vol = metric_data[title]
        metric_sl = get_slice(metric_vol, axis=2, idx=axi_idx)

        # Crop metric slice
        metric_sl = apply_crop(metric_sl, crop_bounds)

        vmin, vmax = metric_ranges[title]

        # Copy the colormap and set NaN/masked areas to black
        cmap = plt.get_cmap(metric_cmaps[title]).copy()
        cmap.set_bad(color="black")

        im = ax.imshow(
            metric_sl,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        # Make colorbar readable on black background
        cbar.ax.tick_params(
            labelsize=7,
            colors="white"
        )
        cbar.outline.set_edgecolor("white")

    elif panel_type == "roi":
        b0_sl = get_slice(b0, axis=2, idx=axi_idx)

        # Crop B0 slice
        b0_sl = apply_crop(b0_sl, crop_bounds)

        ax.imshow(
            b0_sl,
            cmap="gray",
            vmin=b0_vmin,
            vmax=b0_vmax,
            interpolation="nearest"
        )

        for roi_name, roi_vol in roi_data.items():
            roi_sl = get_slice(roi_vol, axis=2, idx=axi_idx)

            # Crop ROI slice using same bounds
            roi_sl = apply_crop(roi_sl, crop_bounds)

            roi_rgba = make_rgba_overlay(
                roi_sl,
                roi_colors[roi_name],
                alpha=0.45
            )

            ax.imshow(
                roi_rgba,
                interpolation="nearest"
            )

    else:
        raise ValueError(f"Unknown panel type: {panel_type}")

    ax.set_title(
        title,
        fontsize=12,
        color="white"
    )
    ax.axis("off")

fig.suptitle(
    f"Axial Metric and ROI Panel, Slice {axi_idx}",
    fontsize=14,
    color="white"
)

fig.savefig(
    panel_out,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.close(fig)

print(f"Saved panel figure to: {panel_out}")
print("Done.")