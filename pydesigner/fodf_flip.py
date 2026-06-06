import itertools
import os
import os.path as op
import numpy as np

from pydesigner.fitting.dwipy import DWI
from pydesigner.system.utils import writeNii

basePath = os.path.join('/Volumes','Flashy','HIE_FBI_003','FBWM_b4000')

dwi_path = os.path.join(basePath, "dwi_preprocessed.nii")
bvec_path = os.path.join(basePath,"dwi_preprocessed.bvec")
bval_path = os.path.join(basePath,"dwi_preprocessed.bval")
mask_path = os.path.join(basePath,"brain_mask.nii")

out_dir = os.path.join(basePath, "debug_fodf_flips")
os.makedirs(out_dir, exist_ok=True)


def transform_gradients(grad, perm=(0, 1, 2), flips=(1, 1, 1)):
    grad = grad.copy()
    g = grad[:, :3].copy()
    g = g[:, perm]
    g *= np.asarray(flips, dtype=float)[None, :]
    grad[:, :3] = g
    return grad


perms = list(itertools.permutations((0, 1, 2)))
flips_list = list(itertools.product((1, -1), repeat=3))

for perm in perms:
    for flips in flips_list:
        label = (
            f"perm{perm[0]}{perm[1]}{perm[2]}"
            f"_flip{flips[0]}{flips[1]}{flips[2]}"
        )
        label = label.replace("-", "m")

        print(f"\nRunning {label}: perm={perm}, flips={flips}")

        dwi = DWI(
            imPath=dwi_path,
            bvecPath=bvec_path,
            bvalPath=bval_path,
            mask=mask_path,
            nthreads=1,
            bvec_flips=(1, 1, 1),
        )

        dwi.grad = transform_gradients(dwi.grad, perm=perm, flips=flips)

        result = dwi.fbi(
            l_max=6,
            fbwm=False,
            rectify=False,
            res="low",
        )

        # Current fbi() returns 12 outputs in your branch:
        (
            zeta,
            faa,
            fodf,
            fodf_mrtrix,
            awf,
            Da,
            De_mean,
            De_ax,
            De_rad,
            De_fa,
            min_cost,
            min_cost_fn,
        ) = result

        writeNii(
            np.real(fodf),
            dwi.hdr,
            op.join(out_dir, f"fbi_odf_raw_{label}.nii"),
        )

        writeNii(
            np.real(fodf_mrtrix),
            dwi.hdr,
            op.join(out_dir, f"fbi_odf_mrtrix_{label}.nii"),
        )

        # writeNii(
        #     np.real(faa),
        #     dwi.hdr,
        #     op.join(out_dir, f"faa_{label}.nii.gz"),
        # )