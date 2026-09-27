"""FastAPI application: composition, upload admission before multipart parsing, headers, logging and static UI."""
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

from wine_scanner.adapters.assets import AssetPack
from wine_scanner.adapters.backend import RecognitionBackend
from wine_scanner.api import create_router, failure
from wine_scanner.application import ScanService
from wine_scanner.contracts import MAX_BYTES, MULTIPART_SLACK, Rejected

STATIC = Path(__file__).resolve().parent / 'static'
UPLOADS = {'/api/recognize': 'recognize', '/v1/eval/predict': 'predict'}
CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
log = logging.getLogger('uvicorn.error')


def create_app(settings):
    # Keep bounded uploads in memory: the default 1 MiB spool would write photos to a temp file.
    MultiPartParser.spool_max_size = MAX_BYTES + MULTIPART_SLACK
    assets = AssetPack(settings.assets_dir, settings.assets_manifest_sha256, settings.expected_profile)
    backend = RecognitionBackend(settings.backend, settings.timeout_s)
    service = ScanService(backend, assets, settings.expected_profile)

    @asynccontextmanager
    async def lifespan(_):
        yield
        await backend.close()

    app = FastAPI(title='Wine scanner gateway', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return failure(Rejected(422, 'bad_request', 'Нужна форма multipart с полем image'))

    @app.middleware('http')
    async def guarded(request: Request, call_next):
        event = UPLOADS.get(request.url.path) if request.method == 'POST' else None
        started, status = time.perf_counter(), 500
        try:
            response = admit(request) if event else None
            if response is None:
                response = await call_next(request)
            status = response.status_code
            response.headers['Content-Security-Policy'] = CSP
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['Referrer-Policy'] = 'no-referrer'
            response.headers.setdefault('Cache-Control', 'no-store')
            return response
        finally:
            if event:
                log.info(json.dumps({'event': event, 'status': status,
                                     'duration_ms': round((time.perf_counter() - started) * 1000)}))

    def admit(request):
        length = request.headers.get('content-length', '')
        if not length.isdigit():
            return failure(Rejected(411, 'length_required', 'Нужен запрос с указанной длиной'))
        if int(length) > MAX_BYTES + MULTIPART_SLACK:
            return failure(Rejected(413, 'too_large', 'Файл больше 20 МиБ'))
        return None

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-cache'})

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    app.include_router(create_router(service))
    return app
