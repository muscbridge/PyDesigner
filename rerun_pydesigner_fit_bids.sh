#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Batch rerun PyDesigner fitting-only script over BIDS subjects
#
# Expected layout from the original run_pydesigner_bids.sh:
#   BIDS_ROOT/
#     sub-*/dwi/*_dir-AP_dwi.nii.gz
#     sub-*/dwi/*_dir-PA_dwi.nii.gz
#     derivatives/dki/sub-*/dwi_preprocessed.nii[.gz]
#     derivatives/dki/sub-*/dwi_preprocessed.bvec
#     derivatives/dki/sub-*/dwi_preprocessed.bval
#     derivatives/dki/sub-*/brain_mask.nii[.gz]
#
# The new run_pydesigner_fit.py is then called on each subject's
# preprocessed PyDesigner files.
# ============================================================

# -----------------------------
# User settings
# -----------------------------

# Run from the BIDS root, or pass BIDS root as first argument:
#   ./rerun_pydesigner_fit_bids.sh /path/to/bids_root
BIDS_ROOT="${1:-$(pwd)}"
DERIV_ROOT="${DERIV_ROOT:-${BIDS_ROOT}/derivatives/dki}"

# Path to your updated fitting-only Python script.
# Edit this or export FIT_SCRIPT=/path/to/run_pydesigner_fit.py before running.
FIT_SCRIPT="${FIT_SCRIPT:-${BIDS_ROOT}/run_pydesigner_fit.py}"
PYTHON_CMD="${PYTHON_CMD:-python}"

# New output subdirectory under each subject's derivatives/dki/sub-* folder.
# Using a new folder avoids overwriting the original PyDesigner outputs.
METRICS_DIR_NAME="${METRICS_DIR_NAME:-metrics_refit}"

# Fit options for run_pydesigner_fit.py
NTHREADS="${NTHREADS:-8}"
RES="${RES:-med}"
LMAX_FBI="${LMAX_FBI:-6}"
BVEC_FLIP_X="${BVEC_FLIP_X:-1}"
BVEC_FLIP_Y="${BVEC_FLIP_Y:--1}"
BVEC_FLIP_Z="${BVEC_FLIP_Z:-1}"

# Set these to 0 if you do not want them passed.
DO_RECTIFY="${DO_RECTIFY:-1}"
DO_FBWM="${DO_FBWM:-1}"

# Set DRY_RUN=1 to print commands without executing.
DRY_RUN="${DRY_RUN:-0}"

# Set SKIP_IF_DONE=1 to skip subjects with existing key outputs.
SKIP_IF_DONE="${SKIP_IF_DONE:-0}"

# -----------------------------
# Helpers
# -----------------------------

find_first() {
    local root="$1"
    shift

    local pattern
    local hit
    for pattern in "$@"; do
        hit="$(find "${root}" -type f -name "${pattern}" -print | head -n 1 || true)"
        if [[ -n "${hit}" ]]; then
            echo "${hit}"
            return 0
        fi
    done

    return 1
}

require_file() {
    local label="$1"
    local file="$2"
    if [[ -z "${file}" || ! -f "${file}" ]]; then
        echo "[MISSING] ${label}: ${file:-not found}"
        return 1
    fi
    return 0
}

# -----------------------------
# Checks
# -----------------------------

if [[ ! -d "${BIDS_ROOT}" ]]; then
    echo "[ERROR] BIDS_ROOT does not exist: ${BIDS_ROOT}"
    exit 1
fi

if [[ ! -d "${DERIV_ROOT}" ]]; then
    echo "[ERROR] DERIV_ROOT does not exist: ${DERIV_ROOT}"
    echo "        Did the original PyDesigner BIDS run finish?"
    exit 1
fi

if [[ ! -f "${FIT_SCRIPT}" ]]; then
    echo "[ERROR] FIT_SCRIPT does not exist: ${FIT_SCRIPT}"
    echo "        Edit FIT_SCRIPT in this script or export FIT_SCRIPT=/path/to/run_pydesigner_fit.py"
    exit 1
fi

RECTIFY_FLAG=()
FBWM_FLAG=()
if [[ "${DO_RECTIFY}" == "1" ]]; then
    RECTIFY_FLAG=(--rectify)
fi
if [[ "${DO_FBWM}" == "1" ]]; then
    FBWM_FLAG=(--fbwm)
fi

# -----------------------------
# Main loop
# -----------------------------

for SUB_DIR in "${BIDS_ROOT}"/sub-*; do
    if [[ ! -d "${SUB_DIR}" ]]; then
        continue
    fi

    SUB_ID="$(basename "${SUB_DIR}")"
    RAW_DWI_DIR="${SUB_DIR}/dwi"
    PYD_DIR="${DERIV_ROOT}/${SUB_ID}"
    OUT_DIR="${PYD_DIR}/${METRICS_DIR_NAME}"
    LOG_DIR="${PYD_DIR}/logs"
    LOG_FILE="${LOG_DIR}/fit_rerun_${SUB_ID}.log"

    echo "============================================================"
    echo "Subject: ${SUB_ID}"
    echo "Raw DWI dir        : ${RAW_DWI_DIR}"
    echo "PyDesigner deriv dir: ${PYD_DIR}"
    echo "Refit output      : ${OUT_DIR}"
    echo "============================================================"

    if [[ ! -d "${RAW_DWI_DIR}" ]]; then
        echo "[SKIP] Missing raw BIDS DWI directory: ${RAW_DWI_DIR}"
        echo
        continue
    fi

    if [[ ! -d "${PYD_DIR}" ]]; then
        echo "[SKIP] Missing PyDesigner derivative directory: ${PYD_DIR}"
        echo
        continue
    fi

    DWI="$(find_first "${PYD_DIR}" "dwi_preprocessed.nii" "dwi_preprocessed.nii.gz" || true)"
    BVEC="$(find_first "${PYD_DIR}" "dwi_preprocessed.bvec" || true)"
    BVAL="$(find_first "${PYD_DIR}" "dwi_preprocessed.bval" || true)"
    MASK="$(find_first "${PYD_DIR}" "brain_mask.nii" "brain_mask.nii.gz" || true)"

    missing=0
    require_file "preprocessed DWI" "${DWI}" || missing=1
    require_file "preprocessed bvec" "${BVEC}" || missing=1
    require_file "preprocessed bval" "${BVAL}" || missing=1
    require_file "brain mask" "${MASK}" || missing=1
    if [[ "${missing}" == "1" ]]; then
        echo "[SKIP] Required fitting inputs not found for ${SUB_ID}"
        echo
        continue
    fi

    mkdir -p "${OUT_DIR}" "${LOG_DIR}"

    if [[ "${SKIP_IF_DONE}" == "1" ]]; then
        if [[ -f "${OUT_DIR}/dti_fa.nii" || -f "${OUT_DIR}/dki_mk.nii" || -f "${OUT_DIR}/fbi_faa.nii" ]]; then
            echo "[SKIP] Existing refit outputs found in ${OUT_DIR}"
            echo
            continue
        fi
    fi

    CMD=(
        "${PYTHON_CMD}" "${FIT_SCRIPT}"
        --dwi "${DWI}"
        --bvec "${BVEC}"
        --bval "${BVAL}"
        --mask "${MASK}"
        --out "${OUT_DIR}"
        --nthreads "${NTHREADS}"
        --res "${RES}"
        --lmax-fbi "${LMAX_FBI}"
        "${RECTIFY_FLAG[@]}"
        "${FBWM_FLAG[@]}"
        --bvec-flips "${BVEC_FLIP_X}" "${BVEC_FLIP_Y}" "${BVEC_FLIP_Z}"
    )

    echo "[RUN] Fitting-only PyDesigner rerun for ${SUB_ID}"
    echo "[LOG] ${LOG_FILE}"
    echo "Command:"
    printf '  %q' "${CMD[@]}"
    echo

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY RUN] Command not executed."
        echo
        continue
    fi

    {
        echo "Started: $(date)"
        echo "Subject: ${SUB_ID}"
        echo "DWI : ${DWI}"
        echo "BVEC: ${BVEC}"
        echo "BVAL: ${BVAL}"
        echo "MASK: ${MASK}"
        echo "OUT : ${OUT_DIR}"
        echo "Command:"
        printf '  %q' "${CMD[@]}"
        echo
        echo

        "${CMD[@]}"

        echo
        echo "Finished: $(date)"
    } 2>&1 | tee "${LOG_FILE}"

    echo "[DONE] ${SUB_ID}"
    echo
done

echo "All available subjects processed."
