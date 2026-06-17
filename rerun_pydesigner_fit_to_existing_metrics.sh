#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Batch rerun PyDesigner fitting-only script over existing
# BIDS derivatives/dki subject folders.
#
# Expected layout:
#   BIDS_ROOT/
#     derivatives/dki/sub-*/
#       dwi_preprocessed.nii[.gz]
#       dwi_preprocessed.bvec
#       dwi_preprocessed.bval
#       brain_mask.nii[.gz]
#       metrics/                 <-- output goes here
#
# You can run this from the BIDS root:
#   ./rerun_pydesigner_fit_bids.sh /path/to/BIDS_ROOT
#
# Or directly from /path/to/BIDS_ROOT/derivatives/dki:
#   ./rerun_pydesigner_fit_bids.sh /path/to/BIDS_ROOT/derivatives/dki
# ============================================================

# -----------------------------
# User settings
# -----------------------------

ROOT_IN="${1:-$(pwd)}"

# If the input path itself looks like derivatives/dki, use it directly.
# Otherwise assume the input is the BIDS root.
if [[ -d "${ROOT_IN}/sub-001" || -n "$(find "${ROOT_IN}" -maxdepth 1 -type d -name 'sub-*' -print -quit 2>/dev/null || true)" ]]; then
    DERIV_ROOT="${DERIV_ROOT:-${ROOT_IN}}"
    BIDS_ROOT="$(cd "${DERIV_ROOT}/../.." && pwd)"
else
    BIDS_ROOT="${ROOT_IN}"
    DERIV_ROOT="${DERIV_ROOT:-${BIDS_ROOT}/derivatives/dki}"
fi

# Path to your updated fitting-only Python script.
# Edit this or export FIT_SCRIPT=/path/to/run_pydesigner_fit.py before running.
FIT_SCRIPT="${FIT_SCRIPT:-${BIDS_ROOT}/run_pydesigner_fit.py}"
PYTHON_CMD="${PYTHON_CMD:-python}"

# Output metrics directory name under each derivatives/dki/sub-* folder.
# Per your request, this writes into the existing metrics folder.
METRICS_DIR_NAME="${METRICS_DIR_NAME:-metrics}"

# By default, require that the metrics folder already exists.
# Set CREATE_METRICS_IF_MISSING=1 to create it when absent.
CREATE_METRICS_IF_MISSING="${CREATE_METRICS_IF_MISSING:-0}"

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

# Set SKIP_IF_DONE=1 to skip subjects with existing key outputs in metrics/.
# Default is 0 because this script is intended to rerun/overwrite metrics.
SKIP_IF_DONE="${SKIP_IF_DONE:-0}"

# Logs are kept outside metrics/ by default.
LOG_DIR_NAME="${LOG_DIR_NAME:-logs}"

# -----------------------------
# Helpers
# -----------------------------

find_first() {
    local root="$1"
    shift

    local pattern
    local hit
    for pattern in "$@"; do
        hit="$(find "${root}" -type f -name "${pattern}" \
            -not -path "*/${METRICS_DIR_NAME}/*" \
            -not -path "*/${LOG_DIR_NAME}/*" \
            -print | head -n 1 || true)"
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

if [[ ! -d "${DERIV_ROOT}" ]]; then
    echo "[ERROR] DERIV_ROOT does not exist: ${DERIV_ROOT}"
    echo "        Expected a BIDS derivatives folder like: BIDS_ROOT/derivatives/dki"
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
# Main loop over derivatives/dki/sub-*
# -----------------------------

echo "BIDS_ROOT : ${BIDS_ROOT}"
echo "DERIV_ROOT: ${DERIV_ROOT}"
echo "FIT_SCRIPT: ${FIT_SCRIPT}"
echo "Output dir: derivatives/dki/sub-*/${METRICS_DIR_NAME}"
echo

for PYD_DIR in "${DERIV_ROOT}"/sub-*; do
    if [[ ! -d "${PYD_DIR}" ]]; then
        continue
    fi

    SUB_ID="$(basename "${PYD_DIR}")"
    OUT_DIR="${PYD_DIR}/${METRICS_DIR_NAME}"
    LOG_DIR="${PYD_DIR}/${LOG_DIR_NAME}"
    LOG_FILE="${LOG_DIR}/fit_rerun_${SUB_ID}.log"

    echo "============================================================"
    echo "Subject             : ${SUB_ID}"
    echo "PyDesigner deriv dir: ${PYD_DIR}"
    echo "Metrics output      : ${OUT_DIR}"
    echo "============================================================"

    if [[ ! -d "${OUT_DIR}" ]]; then
        if [[ "${CREATE_METRICS_IF_MISSING}" == "1" ]]; then
            mkdir -p "${OUT_DIR}"
            echo "[INFO] Created missing metrics directory: ${OUT_DIR}"
        else
            echo "[SKIP] Missing existing metrics directory: ${OUT_DIR}"
            echo "       Set CREATE_METRICS_IF_MISSING=1 to create it automatically."
            echo
            continue
        fi
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

    mkdir -p "${LOG_DIR}"

    if [[ "${SKIP_IF_DONE}" == "1" ]]; then
        if [[ -f "${OUT_DIR}/dti_fa.nii" || -f "${OUT_DIR}/dki_mk.nii" || -f "${OUT_DIR}/fbi_faa.nii" ]]; then
            echo "[SKIP] Existing outputs found in ${OUT_DIR}"
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

echo "All available derivatives/dki subjects processed."
