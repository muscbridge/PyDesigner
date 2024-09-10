FROM dmri/ci-cd-py3.11:e93dd885
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /src
COPY . /src
ENV TEST_DIR=/results

RUN uv pip install -rrequirements.txt && \
    uv pip install -rrequirements-dev.txt && \
    uv pip install --no-deps -e.
COPY ./tests/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
