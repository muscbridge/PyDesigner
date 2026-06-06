import os
import numpy as np
import nibabel as nib

# Adjust this import to match your local PyDesigner structure
from pydesigner.fitting.dwipy import DWI
from pydesigner.system.utils import writeNii

basePath = os.path.join('/Volumes','Flashy','HIE_FBI_003','FBWM_b4000')

dwi_path = os.path.join(basePath, "dwi_preprocessed.nii")
bvec_path = os.path.join(basePath,"dwi_preprocessed.bvec")
bval_path = os.path.join(basePath,"dwi_preprocessed.bval")
mask_path = os.path.join(basePath,"brain_mask.nii")

out_dir = os.path.join(basePath, "debug_fodf_flips")
os.makedirs(out_dir, exist_ok=True)


def apply_gradient_flip(dwi_obj, flips):
    """
    flips = (fx, fy, fz), each either +1 or -1.
    """
    flips = np.asarray(flips, dtype=float)
    dwi_obj.grad[:, :3] = dwi_obj.grad[:, :3] * flips[None, :]
    return dwi_obj


flip_tests = {
    "none":      ( 1,  1,  1),
    "xflip":     (-1,  1,  1),
    "yflip":     ( 1, -1,  1),
    "zflip":     ( 1,  1, -1),
    "xyflip":    (-1, -1,  1),
    "xzflip":    (-1,  1, -1),
    "yzflip":    ( 1, -1, -1),
    "xyzflip":   (-1, -1, -1),
}


for label, flips in flip_tests.items():
    print(f"\nRunning flip test: {label} {flips}")

    dwi = DWI(
        imPath=dwi_path,
        bvecPath=bvec_path,
        bvalPath=bval_path,
        mask=mask_path,
        nthreads=1,
    )

    dwi = apply_gradient_flip(dwi, flips)

    # FBI/fODF only. Do not run full FBWM.
    zeta, faa, fodf, fodf_mrtrix, min_awf, Da, De_mean, De_ax, De_rad, De_fa, min_cost, min_cost_fn = dwi.fbi(
        l_max=6,
        fbwm=False,
        rectify=True,
        res="low",
    )

    # Save the fODF SH coefficients or fODF output returned by dwi.fbi().
    # Depending on your downstream viewer, this may be the file to inspect.
    out_path = os.path.join(out_dir, f"fodf_{label}.nii")
    writeNii(np.asarray(fodf).real, dwi.hdr, out_path)
    
    writeNii(
        np.asarray(fodf_mrtrix).real,
        dwi.hdr,
        os.path.join(out_dir, f"fodf_mrtrix_{label}.nii"),
    )

    # # Save FAA too as a scalar QC map.
    # faa_path = os.path.join(out_dir, f"faa_{label}.nii.gz")
    # writeNii(np.asarray(faa), dwi.hdr, faa_path)

    print(f"Saved: {out_path}")
    # print(f"Saved: {faa_path}")