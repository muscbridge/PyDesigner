FROM dmri/ci-cd-py3.11:e93dd885
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /src
COPY . /src
ENV TEST_DIR=/tests

RUN pip install uv && \
    uv venv ${VIRTUAL_ENV} && \
    uv pip install -U pip setuptools wheel
RUN uv pip install -rrequirements.txt -rrequirements-dev.txt
RUN uv pip install --no-deps -e.
RUN mkdir -p ${TEST_DIR}
COPY ./tests/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
