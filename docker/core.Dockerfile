# Recognition core for Linux CPU: vendored recognizer + PP-OCRv6 adapter. Weights, gallery, catalogue and profiles
# come from the external asset pack, mounted read-only into /srv/core by compose.yaml (profile "linux").
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/core-requirements.txt /tmp/core-requirements.txt
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/core-requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ && rm /tmp/core-requirements.txt
# The pinned LightGlue distribution only satisfies a vendored version guard and is never imported, so its own
# dependencies are deliberately absent. pip check must report exactly these three lines; anything else fails.
RUN python -c "import subprocess, sys; \
d = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True); \
allowed = ['lightglue 0.0 requires kornia, which is not installed.', \
           'lightglue 0.0 requires matplotlib, which is not installed.', \
           'lightglue 0.0 requires opencv-python, which is not installed.']; \
ok = d.returncode in (0, 1) and not d.stderr.strip() and sorted(d.stdout.splitlines()) == allowed; \
sys.exit(0 if ok else 'pip check: exit %d: %s %s' % (d.returncode, d.stdout, d.stderr))"
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin scanner
COPY recognizer/ /srv/core/
COPY wine_scanner_core/ /srv/app/wine_scanner_core/
ENV PYTHONPATH=/srv/app:/srv/core HOME=/tmp HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True OMP_NUM_THREADS=2 \
    WINE_SCANNER_CORE_ROOT=/srv/core WINE_SCANNER_CORE_MANIFEST=/srv/pack/manifest.json \
    WINE_SCANNER_CORE_HOST=0.0.0.0 WINE_SCANNER_CORE_PORT=8375
WORKDIR /srv/core
USER 10001:10001
EXPOSE 8375
# Loading verifies every asset and builds the full recognizer: about 5 minutes on 2 CPUs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8375/health/ready', timeout=3)"]
ENTRYPOINT ["python", "-X", "faulthandler", "-m", "wine_scanner_core"]
CMD ["serve"]
