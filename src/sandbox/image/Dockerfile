FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/workspace/tmp \
    TMP=/workspace/tmp \
    TEMP=/workspace/tmp \
    HOME=/workspace/tmp

RUN useradd --uid 1000 --no-create-home --shell /bin/bash sandbox \
    && mkdir -p /workspace/input /workspace/scripts /workspace/output /workspace/tmp \
    && /bin/bash --version

WORKDIR /workspace
USER sandbox
CMD ["/usr/local/bin/python", "--version"]
