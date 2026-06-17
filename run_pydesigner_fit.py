#!/usr/bin/env python

import argparse
import os
import os.path as op
import numpy as np
import glob
import nibabel as nib

from pydesigner.fitting.dwipy import DWI
from pydesigner.system.utils import writeNii
from pydesigner.tractography import odf



'''
python run_pydesigner_fit.py \
  --dwi /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/dwi_preprocessed.nii \
  --bvec /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/dwi_preprocessed.bvec \
  --bval /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/dwi_preprocessed.bval \
  --mask /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/brain_mask_eroF_noCSF.nii \
  --out /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/metrics \
  --nthreads 8 \
  --res high \
  --lmax-fbi 6 \
  --rectify \
  --fbwm \
  --bvec-flips 1 -1 1

'''
def transform_gradients(grad, perm=(0, 1, 2), flips=(1, 1, 1)):
        """
        Apply axis permutation and sign flips to gradient table.

        grad shape: (N, 4), columns [gx, gy, gz, bval]
        perm: axis order, e.g. (0, 1, 2), (0, 2, 1), etc.
        flips: signs after permutation, e.g. (1, -1, 1)
        """
        grad = grad.copy()

        g = grad[:, :3].copy()
        g = g[:, perm]
        g *= np.asarray(flips, dtype=float)[None, :]

        grad[:, :3] = g
        return grad

def parse_flips(values):
    flips = tuple(float(v) for v in values)
    if len(flips) != 3:
        raise ValueError("--bvec-flips must have exactly three values, e.g. 1 -1 1")
    if any(v not in (-1.0, 1.0) for v in flips):
        raise ValueError("--bvec-flips values must be ±1")
    return flips
def create_combined_jhu_wm_mask(
    jhu_mask_dir,
    reference_nii,
    out_path,
    pattern="jhu_to_fa_*_mask.nii*",
):
    """
    Combine all JHU ROI masks into one binary WM mask.

    Parameters
    ----------
    jhu_mask_dir : str
        Directory containing files like jhu_to_fa_*_mask.nii.
    reference_nii : str
        Reference image defining output shape/affine, usually dwi_preprocessed.nii.
    out_path : str
        Output path for combined WM mask.
    pattern : str
        Glob pattern for JHU mask files.

    Returns
    -------
    wm_mask : ndarray bool
        Combined binary WM mask.
    out_path : str
        Path to saved mask.
    """
    mask_paths = sorted(glob.glob(op.join(jhu_mask_dir, pattern)))

    if len(mask_paths) == 0:
        raise FileNotFoundError(
            f"No JHU mask files found in {jhu_mask_dir} with pattern {pattern}"
        )

    ref_img = nib.load(reference_nii)
    ref_shape = ref_img.shape[:3]
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    wm_mask = np.zeros(ref_shape, dtype=bool)

    print("\nCreating combined JHU WM mask...")
    print(f"JHU mask dir: {jhu_mask_dir}")
    print(f"Pattern: {pattern}")
    print(f"Reference shape: {ref_shape}")
    print(f"Number of masks found: {len(mask_paths)}")

    for mask_path in mask_paths:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()

        if mask_data.ndim > 3:
            mask_data = mask_data[..., 0]

        if mask_data.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for {mask_path}: "
                f"mask shape {mask_data.shape}, reference shape {ref_shape}"
            )

        if not np.allclose(mask_img.affine, ref_affine, atol=1e-3):
            print(
                f"WARNING: affine mismatch for {op.basename(mask_path)}. "
                "Shape matches, but affine differs slightly."
            )

        this_mask = np.isfinite(mask_data) & (mask_data > 0)
        nvox = int(this_mask.sum())

        print(f"  {op.basename(mask_path)}: {nvox} voxels")

        wm_mask |= this_mask

    out_header = ref_header.copy()
    out_header.set_data_dtype(np.uint8)
    out_header.set_data_shape(ref_shape)

    out_img = nib.Nifti1Image(wm_mask.astype(np.uint8), ref_affine, out_header)
    out_img.set_qform(ref_affine, code=1)
    out_img.set_sform(ref_affine, code=1)

    nib.save(out_img, out_path)

    print(f"Combined JHU WM mask voxels: {int(wm_mask.sum())}")
    print(f"Wrote combined JHU WM mask: {out_path}")

    return wm_mask, out_path


def apply_spatial_mask(data, mask, fill_value=0):
    """
    Apply a 3D spatial mask to a 3D or 4D image array.

    For 4D data, the 3D mask is applied to every volume/coefficient.
    """
    if data is None:
        return None

    arr = np.asarray(data).copy()
    mask = np.asarray(mask).astype(bool)

    if arr.shape[:3] != mask.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data spatial shape {arr.shape[:3]}"
        )

    if arr.ndim == 3:
        arr[~mask] = fill_value
    elif arr.ndim == 4:
        arr[~mask, ...] = fill_value
    else:
        raise ValueError(
            f"apply_spatial_mask only supports 3D/4D arrays. Got shape {arr.shape}"
        )

    return arr

def yflip_dt_for_odf(dt):
    """
    Apply y-axis coordinate flip to DT components for ODF/SH export only.

    Expected DT component order used by odf.odfmodel:
        [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]

    For y -> -y:
        Dxy flips sign
        Dyz flips sign
        Dxx, Dyy, Dzz, Dxz unchanged
    """
    dt = np.asarray(dt).copy()

    if dt.shape[-1] != 6:
        raise ValueError(f"Expected DT last dimension = 6, got shape {dt.shape}")

    signs = np.array([1, 1, 1, -1, 1, -1], dtype=dt.dtype)

    return dt * signs.reshape((1, 1, 1, 6))


def yflip_kt_for_odf(kt):
    """
    Apply y-axis coordinate flip to KT components for ODF/SH export only.

    Expected KT component order used by odf.odfmodel:
        0  Wxxxx
        1  Wyyyy
        2  Wzzzz
        3  Wxxxy
        4  Wxxxz
        5  Wxyyy
        6  Wxzzz
        7  Wyyyz
        8  Wyzzz
        9  Wxxyy
        10 Wxxzz
        11 Wyyzz
        12 Wxxyz
        13 Wxyyz
        14 Wxyzz

    For y -> -y, components with an odd number of y indices flip sign.
    """
    kt = np.asarray(kt).copy()

    if kt.shape[-1] != 15:
        raise ValueError(f"Expected KT last dimension = 15, got shape {kt.shape}")

    signs = np.array(
        [
            1,   # Wxxxx
            1,   # Wyyyy
            1,   # Wzzzz
            -1,  # Wxxxy
            1,   # Wxxxz
            -1,  # Wxyyy
            1,   # Wxzzz
            -1,  # Wyyyz
            -1,  # Wyzzz
            1,   # Wxxyy
            1,   # Wxxzz
            1,   # Wyyzz
            -1,  # Wxxyz
            1,   # Wxyyz
            -1,  # Wxyzz
        ],
        dtype=kt.dtype,
    )

    return kt * signs.reshape((1, 1, 1, 15))

def main():
    parser = argparse.ArgumentParser(
        description="Run PyDesigner fitting only, starting from dwi_preprocessed.nii."
    )

    parser.add_argument("--dwi", required=True, help="Path to dwi_preprocessed.nii or .nii.gz")
    parser.add_argument("--bvec", required=True, help="Path to dwi_preprocessed.bvec")
    parser.add_argument("--bval", required=True, help="Path to dwi_preprocessed.bval")
    parser.add_argument("--mask", required=True, help="Path to brain_mask.nii or .nii.gz")
    parser.add_argument("--out", required=True, help="Output directory for metrics")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--res", default="med", choices=["low", "med", "high"])
    parser.add_argument("--lmax-fbi", type=int, default=6)
    parser.add_argument("--rectify", action="store_true", help="Rectify FBI fODF")
    parser.add_argument("--fbwm", action="store_true", help="Run FBWM metrics if FBI + DKI are available")
    parser.add_argument(
        "--bvec-flips",
        nargs=3,
        default=(1, 1, 1),
        help="Gradient sign flips for x y z. Default is 1 -1 1 based on your y-flip test.",
    )
    parser.add_argument(
        "--jhu-mask-dir",
        default=None,
        help="Directory containing jhu_to_fa_*_mask.nii files to combine into a WM mask.",
    )
    parser.add_argument(
        "--jhu-mask-pattern",
        default="jhu_to_fa_*_mask.nii*",
        help="Glob pattern for JHU ROI masks. Default: jhu_to_fa_*_mask.nii*",
    )
    parser.add_argument(
        "--wm-mask-out",
        default=None,
        help="Output path for combined JHU WM mask. Default: <out>/jhu_wm_mask.nii.gz",
    )

    args = parser.parse_args()
    flips = parse_flips(args.bvec_flips)

    os.makedirs(args.out, exist_ok=True)
    
    jhu_wm_mask = None
    jhu_wm_mask_path = None

    if args.jhu_mask_dir is not None:
        jhu_wm_mask_path = (
            args.wm_mask_out
            if args.wm_mask_out is not None
            else op.join(args.out, "jhu_wm_mask.nii.gz")
            
        )

        jhu_wm_mask, jhu_wm_mask_path = create_combined_jhu_wm_mask(
            jhu_mask_dir=args.jhu_mask_dir,
            reference_nii=args.dwi,
            out_path=jhu_wm_mask_path,
            pattern=args.jhu_mask_pattern,
        )

    print("\nLoading preprocessed DWI...")
    img = DWI(
        imPath=args.dwi,
        bvecPath=args.bvec,
        bvalPath=args.bval,
        mask=args.mask,
        nthreads=args.nthreads,
        bvec_flips=tuple(args.bvec_flips),
    )

    # img.grad = transform_gradients(
    #     img.grad,
    #     perm=(0, 1, 2),
    #     flips=(1, 1, 1),
    # )

    print("Detected protocols:", img.tensorType())

    print("\nFitting tensor model...")
    img.fit(constraints=None)

    tensor_type = "dki" if img.isdki() else "dti"
    print(f"Tensor type used for tensorReorder(): {tensor_type}")

    DT, KT = img.tensorReorder(tensor_type)

    dt_path = op.join(args.out, "DT.nii")
    kt_path = op.join(args.out, "KT.nii")

    writeNii(DT, img.hdr, dt_path)
    print(f"Wrote {dt_path}")
    
    # ------------------------------------------------------------
    # Create ODF-coordinate DT/KT copies.
    # These are used only for DTI/DKI ODF SH export, not scalar maps.
    # Validated correction: perm012_flip1m11 = y-axis flip.
    # ------------------------------------------------------------
    DT_odf = yflip_dt_for_odf(DT)
    dt_odf_path = op.join(args.out, "DT_for_odf.nii")
    writeNii(DT_odf, img.hdr, dt_odf_path)
    print(f"Wrote {dt_odf_path}")

    kt_odf_path = None
    
    if tensor_type == "dki":
        writeNii(KT, img.hdr, kt_path)
        print(f"Wrote {kt_path}")

        KT_odf = yflip_kt_for_odf(KT)
        kt_odf_path = op.join(args.out, "KT_for_odf.nii")
        writeNii(KT_odf, img.hdr, kt_odf_path)
        print(f"Wrote {kt_odf_path}")
    else:
        kt_odf_path = None

    # ------------------------------------------------------------
    # DTI metrics + DTI ODF SH
    # ------------------------------------------------------------
    if img.isdti():
        print("\nExtracting DTI metrics...")
        md, rd, ad, fa, fe, trace = img.extractDTI()

        writeNii(md, img.hdr, op.join(args.out, "dti_md.nii"))
        writeNii(rd, img.hdr, op.join(args.out, "dti_rd.nii"))
        writeNii(ad, img.hdr, op.join(args.out, "dti_ad.nii"))
        writeNii(fa, img.hdr, op.join(args.out, "dti_fa.nii"))
        writeNii(fe, img.hdr, op.join(args.out, "dti_fe.nii"))
        writeNii(trace, img.hdr, op.join(args.out, "dti_trace.nii"))

        print("Computing DTI ODF SH...")
        dti_model = odf.odfmodel(
            dt=dt_odf_path,
            mask=args.mask,
            l_max=2,
            res=args.res,
            nthreads=args.nthreads,
        )
        dti_odfs = dti_model.dtiodf()
        dti_sh = dti_model.odf2sh(dti_odfs)
        dti_model.savenii(dti_sh, op.join(args.out, "dti_odf.nii"))

    # ------------------------------------------------------------
    # DKI metrics + DKI ODF SH
    # ------------------------------------------------------------
    if img.isdki():
        print("\nExtracting DKI metrics...")
        mk, rk, ak, kfa, mkt, trace = img.extractDKI()

        writeNii(mk, img.hdr, op.join(args.out, "dki_mk.nii"))
        writeNii(rk, img.hdr, op.join(args.out, "dki_rk.nii"))
        writeNii(ak, img.hdr, op.join(args.out, "dki_ak.nii"))
        writeNii(kfa, img.hdr, op.join(args.out, "dki_kfa.nii"))
        writeNii(mkt, img.hdr, op.join(args.out, "dki_mkt.nii"))
        writeNii(trace, img.hdr, op.join(args.out, "dki_trace.nii"))

        print("Computing DKI ODF SH...")
        dki_model = odf.odfmodel(
            dt=dt_odf_path,
            kt=kt_odf_path,
            mask=args.mask,
            l_max=6,
            res=args.res,
            nthreads=args.nthreads,
        )
        dki_odfs = dki_model.dkiodf(fa_t=0.90)
        dki_sh = dki_model.odf2sh(dki_odfs)
        dki_model.savenii(dki_sh, op.join(args.out, "dki_odf.nii"))

    # ------------------------------------------------------------
    # FBI / FBWM
    # ------------------------------------------------------------
    if img.isfbi():
        print("\nRunning FBI / FBWM...")
        run_fbwm = bool(args.fbwm and img.isfbwm())

        (
            zeta,
            faa,
            sph,
            sph_mrtrix,
            awf,
            Da,
            De_mean,
            De_ax,
            De_rad,
            De_fa,
            min_cost,
            min_cost_fn,
        ) = img.fbi(
            l_max=args.lmax_fbi,
            fbwm=run_fbwm,
            rectify=args.rectify,
            res=args.res,
        )
        
        if jhu_wm_mask is not None:
            print("\nApplying combined JHU WM mask to FBI/FBWM outputs...")

            zeta = apply_spatial_mask(zeta, jhu_wm_mask)
            faa = apply_spatial_mask(faa, jhu_wm_mask)

            sph = apply_spatial_mask(np.real(sph), jhu_wm_mask)
            sph_mrtrix = apply_spatial_mask(np.real(sph_mrtrix), jhu_wm_mask)

            if run_fbwm:
                awf = apply_spatial_mask(awf, jhu_wm_mask)
                Da = apply_spatial_mask(Da, jhu_wm_mask)
                De_mean = apply_spatial_mask(De_mean, jhu_wm_mask)
                De_ax = apply_spatial_mask(De_ax, jhu_wm_mask)
                De_rad = apply_spatial_mask(De_rad, jhu_wm_mask)
                De_fa = apply_spatial_mask(De_fa, jhu_wm_mask)
                min_cost = apply_spatial_mask(min_cost, jhu_wm_mask)
                min_cost_fn = apply_spatial_mask(min_cost_fn, jhu_wm_mask)

        writeNii(zeta, img.hdr, op.join(args.out, "fbi_zeta.nii"))
        writeNii(faa, img.hdr, op.join(args.out, "fbi_faa.nii"))

        # Save both. For MRtrix, use the MRtrix/Tournier-converted version.
        # writeNii(np.real(sph), img.hdr, op.join(args.out, "fbi_odf_raw.nii"))
        writeNii(np.real(sph), img.hdr, op.join(args.out, "fbi_odf.nii"))
        writeNii(np.real(sph_mrtrix), img.hdr, op.join(args.out, "fbi_odf_mrtrix.nii"))

        if run_fbwm:
            writeNii(awf, img.hdr, op.join(args.out, "fbwm_awf.nii"))
            writeNii(Da, img.hdr, op.join(args.out, "fbwm_Da.nii"))
            writeNii(De_mean, img.hdr, op.join(args.out, "fbwm_De_mean.nii"))
            writeNii(De_ax, img.hdr, op.join(args.out, "fbwm_De_ax.nii"))
            writeNii(De_rad, img.hdr, op.join(args.out, "fbwm_De_rad.nii"))
            writeNii(De_fa, img.hdr, op.join(args.out, "fbwm_fae.nii"))
            writeNii(min_cost, img.hdr, op.join(args.out, "fbwm_minCost.nii"))
            writeNii(min_cost_fn, img.hdr, op.join(args.out, "fbwm_minCostFn.nii"))

    print("\nDone.")


if __name__ == "__main__":
    main()