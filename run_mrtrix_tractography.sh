#!/usr/bin/env bash
'''
./run_mrtrix_tractography.sh \
/Volumes/Flashy/HIE_FBI_003/FBWM_b4000/metrics \
/Volumes/Flashy/HIE_FBI_003/FBWM_b4000/wm.nii \
/Volumes/Flashy/HIE_FBI_003/FBWM_b4000/metrics \
SD_STREAM
'''

set -euo pipefail

METRICS_DIR="${1:?Usage: run_mrtrix_tractography.sh METRICS_DIR MASK_NII OUT_DIR [algorithm]}"
MASK_NII="${2:?Usage: run_mrtrix_tractography.sh METRICS_DIR MASK_NII OUT_DIR [algorithm]}"
OUT_DIR="${3:?Usage: run_mrtrix_tractography.sh METRICS_DIR MASK_NII OUT_DIR [algorithm]}"

# Deterministic FOD-based tractography:
ALGORITHM="${4:-sd_stream}"

# To run probabilistic iFOD2 instead:
# ALGORITHM="ifod2"

N_STREAMLINES="${N_STREAMLINES:-100000}"
CUTOFF="${CUTOFF:-0.05}"
ANGLE="${ANGLE:-45}"
MINLENGTH="${MINLENGTH:-10}"
MAXLENGTH="${MAXLENGTH:-250}"
STEP="${STEP:-0}"

mkdir -p "${OUT_DIR}"

echo "MRtrix algorithm: ${ALGORITHM}"
echo "Streamlines: ${N_STREAMLINES}"
echo "Cutoff: ${CUTOFF}"

MASK_MIF="${OUT_DIR}/brain_mask.mif"
mrconvert "${MASK_NII}" "${MASK_MIF}" -force

run_tracking () {
    local name="$1"
    local fod_nii="$2"

    if [[ ! -f "${fod_nii}" ]]; then
        echo "Skipping ${name}: missing ${fod_nii}"
        return 0
    fi

    local fod_mif="${OUT_DIR}/${name}_fod.mif"
    local tck="${OUT_DIR}/${name}_${ALGORITHM}.tck"
    local density="${OUT_DIR}/${name}_${ALGORITHM}_tdi.nii.gz"

    echo ""
    echo "=============================="
    echo "Running ${name}"
    echo "FOD: ${fod_nii}"
    echo "Output: ${tck}"
    echo "=============================="

    mrconvert "${fod_nii}" "${fod_mif}" -force

    TCKGEN_ARGS=(
        "${fod_mif}"
        "${tck}"
        -algorithm "${ALGORITHM}"
        -seed_image "${MASK_MIF}"
        -mask "${MASK_MIF}"
        -select "${N_STREAMLINES}"
        -cutoff "${CUTOFF}"
        -angle "${ANGLE}"
        -minlength "${MINLENGTH}"
        -maxlength "${MAXLENGTH}"
        -force
    )

    # Optional explicit step size.
    # If STEP=0, let MRtrix use its algorithm-specific default.
    if [[ "${STEP}" != "0" ]]; then
        TCKGEN_ARGS+=( -step "${STEP}" )
    fi

    tckgen "${TCKGEN_ARGS[@]}"

    tckinfo "${tck}"

    # Quick streamline-density image for visual QC
    tckmap "${tck}" "${density}" -template "${MASK_MIF}" -force

    echo "Wrote ${tck}"
    echo "Wrote ${density}"
}

run_tracking "dti" "${METRICS_DIR}/dti_odf.nii"
run_tracking "dki" "${METRICS_DIR}/dki_odf.nii"

# Prefer MRtrix-converted FBI ODF if present.
if [[ -f "${METRICS_DIR}/fbi_odf_mrtrix.nii" ]]; then
    run_tracking "fbi" "${METRICS_DIR}/fbi_odf_mrtrix.nii"
else
    run_tracking "fbi" "${METRICS_DIR}/fbi_odf.nii"
fi

echo ""
echo "All tractography complete."