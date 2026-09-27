FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin scanner
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY wine_scanner ./wine_scanner
ENV WINE_SCANNER_HOST=0.0.0.0 \
    WINE_SCANNER_PORT=8287 \
    WINE_SCANNER_ASSETS_DIR=/assets \
    WINE_SCANNER_BACKEND=http://host.docker.internal:8175
USER 10001:10001
EXPOSE 8287
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8287/health/live', timeout=3)"]
ENTRYPOINT ["python", "-m", "wine_scanner"]
