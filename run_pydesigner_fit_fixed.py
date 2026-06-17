#!/usr/bin/env python

import argparse
import os
import os.path as op
import numpy as np

from pydesigner.fitting.dwipy import DWI
from pydesigner.system.utils import writeNii
from pydesigner.tractography import odf

'''
basePath = os.path.join('/Volumes','Flashy','HIE_FBI_003','FBWM_b4000')

python run_pydesigner_fit.py \
  --dwi /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/dwi_preprocessed.nii \
  --bvec /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/dwi_preprocessed.bvec \
  --bval /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/dwi_preprocessed.bval \
  --mask /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/brain_mask.nii \
  --out /Volumes/Flashy/HIE_FBI_003/FBWM_b4000/metrics \
  --nthreads 8 \
  --res med \
  --lmax-fbi 6 \
  --rectify \
  --fbwm \
  --bvec-flips 1 1 1

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
        type=float,
        default=(1.0, -1.0, 1.0),
        help="Gradient sign flips for x y z. Default is 1 -1 1 based on your y-flip test.",
    )

    args = parser.parse_args()
    flips = parse_flips(args.bvec_flips)

    os.makedirs(args.out, exist_ok=True)

    print("\nLoading preprocessed DWI...")
    img = DWI(
        imPath=args.dwi,
        bvecPath=args.bvec,
        bvalPath=args.bval,
        mask=args.mask,
        nthreads=args.nthreads,
        bvec_flips=flips,
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

    if tensor_type == "dki":
        writeNii(KT, img.hdr, kt_path)
        print(f"Wrote {kt_path}")

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
            dt=dt_path,
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
            dt=dt_path,
            kt=kt_path,
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

        writeNii(zeta, img.hdr, op.join(args.out, "fbi_zeta.nii"))
        writeNii(faa, img.hdr, op.join(args.out, "fbi_faa.nii"))

        # Save both. For MRtrix, use the MRtrix/Tournier-converted version.
        # writeNii(np.real(sph), img.hdr, op.join(args.out, "fbi_odf_raw.nii"))
        writeNii(np.real(sph_mrtrix), img.hdr, op.join(args.out, "fbi_odf.nii"))
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