# ==============================================================================
# NeuroDock / PyDesigner v2.0.0
#
# Docker image containing PyDesigner and external neuroimaging dependencies:
# - Python 3.14
# - uv
# - PyDesigner-DWI 2.0.0
# - MRtrix3
# - FSL
#
# NOTE:
# FSL and MRTrix3 are open-source, free software. Users are responsible for complying with their license
# terms.
# ==============================================================================

# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL maintainer="MUSC BRIDGE"
LABEL org.opencontainers.image.title="PyDesigner-DWI"
LABEL org.opencontainers.image.description="A DTI, DKI, FBI, and FBWM diffusion MRI processing pipeline"
LABEL org.opencontainers.image.url="https://github.com/muscbridge/PyDesigner"
LABEL org.opencontainers.image.source="https://github.com/muscbridge/PyDesigner"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="MUSC BRIDGE"

ARG PYDESIGNER_WHEEL=dist/pydesigner_dwi-2.0.0-py3-none-any.whl
ARG FSL_VERSION=6.0.7.18

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV PYDESIGNER_VENV=/opt/pydesigner/.venv
ENV PATH="/opt/pydesigner/.venv/bin:/usr/local/fsl/bin:/usr/lib/mrtrix3/bin:${PATH}"
ENV FSLDIR=/usr/local/fsl
ENV FSLOUTPUTTYPE=NIFTI
ENV PATH="/opt/pydesigner/.venv/bin:/usr/local/fsl/bin:/usr/lib/mrtrix3/bin:${PATH}"

# ------------------------------------------------------------------------------
# System dependencies + MRtrix3
# ------------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        file \
        dc \
        git \
        tar \
        bzip2 \
        mrtrix3 \
        python3 \
        python3-venv \
        libgomp1 \
        libquadmath0 \
        libgtk2.0-0 \
        mesa-utils \
        libgl1 \
        libglib2.0-0 \
        && \
    rm -rf /var/lib/apt/lists/*
# ------------------------------------------------------------------------------
# uv
# ------------------------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ------------------------------------------------------------------------------
# FSL
# ------------------------------------------------------------------------------

RUN wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py \
        -O /tmp/fslinstaller.py && \
    python3 /tmp/fslinstaller.py \
        -d "${FSLDIR}" \
        -V "${FSL_VERSION}" \
        --skip_registration && \
    rm -f /tmp/fslinstaller.py

# ------------------------------------------------------------------------------
# PyDesigner
# ------------------------------------------------------------------------------

WORKDIR /opt/pydesigner

COPY ${PYDESIGNER_WHEEL} /tmp/pydesigner_dwi-2.0.0-py3-none-any.whl

RUN uv python install 3.14 && \
    uv venv "${PYDESIGNER_VENV}" --python 3.14 && \
    uv pip install --python "${PYDESIGNER_VENV}/bin/python" /tmp/pydesigner_dwi-2.0.0-py3-none-any.whl && \
    rm -f /tmp/pydesigner_dwi-2.0.0-py3-none-any.whl

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

RUN cat > /usr/local/bin/pydesigner-entrypoint <<'EOF' && \
    chmod +x /usr/local/bin/pydesigner-entrypoint
#!/usr/bin/env bash
set -e

if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then
    source "${FSLDIR}/etc/fslconf/fsl.sh"
fi

exec pydesigner "$@"
EOF

# ------------------------------------------------------------------------------
# Validate image
# ------------------------------------------------------------------------------

RUN python -V && \
    python -c "import pydesigner; import pydesigner.fitting.dwipy; print('PyDesigner import OK:', pydesigner.__version__)" && \
    mrconvert -version && \
    dwidenoise -version && \
    bash -lc "source ${FSLDIR}/etc/fslconf/fsl.sh && bet --version || true"

# ------------------------------------------------------------------------------
# Non-root runtime user
# ------------------------------------------------------------------------------

RUN useradd -ms /bin/bash bridge && \
    mkdir -p /data && \
    chown -R bridge:bridge /data /opt/pydesigner

USER bridge
WORKDIR /data

ENTRYPOINT ["pydesigner-entrypoint"]
CMD ["--help"]
