# syntax=docker/dockerfile:1

# Stage 1: install dependencies in an isolated virtual environment.
FROM python:3.12-slim-bookworm AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: keep only the runtime files and installed dependencies.
FROM python:3.12-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PORT=8000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder "${VIRTUAL_ENV}" "${VIRTUAL_ENV}"
COPY src/ ./src/
COPY artifacts/ ./artifacts/

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port \"$PORT\""]
