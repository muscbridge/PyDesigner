# ==============================================================================
# NeuroDock
# A docker container that contains all PyDesigner dependencies such as MRTRIX3,
# FSL, and Python to preprocess diffusion MRI images.
#
# Maintainer: Siddhartha Dhiman
# ==============================================================================

# Load base Ubuntu image
FROM dmri/ci-cd-py3.12:6f2600ca AS base
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /src

# Labels
LABEL maintainer="Siddhartha Dhiman (siddhartha.dhiman@gmail.com)"
LABEL org.label-schema.name="dmri/pydesigner"
LABEL org.label-schema.description="A state-of-the-art difusion and kurtosis MRI processing pipeline"
LABEL org.label-schema.url="https://github.com/m-ama/"
LABEL org.label-schema.vcs-url="https://github.com/m-ama/NeuroDock.git"
LABEL org.label-schema.vendor="MUSC BRIDGE"

# Copy and install PyDesigner
FROM base AS dependencies
COPY requirements.txt ./
RUN uv pip install -r requirements.txt

FROM dependencies AS development
COPY requirements-dev.txt ./
RUN uv pip install -rrequirements-dev.txt
COPY . .
RUN uv pip install --no-deps -e.

FROM dependencies AS pyc
COPY . .
RUN python -m compileall -bqj0 .
RUN find . -name "*.py" -not -name "__init__.py" -delete

FROM pyc AS production
COPY --from=pyc /src .
RUN uv pip install --no-deps -e.

RUN useradd -ms /bin/bash bridge
USER bridge
