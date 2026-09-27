"""Public routes: UI status and recognition, reference thumbnails, the evaluation slug and health probes."""
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from wine_scanner.contracts import MAX_BYTES, Rejected, inspect_image, parse_roi


def failure(error):
    headers = {'Retry-After': str(error.retry_after)} if error.retry_after is not None else None
    return JSONResponse(error.body(), status_code=error.status, headers=headers)


def create_router(service):
    router = APIRouter()

    async def upload(image, target_roi):
        data = await image.read(MAX_BYTES + 1)
        info = inspect_image(data, image.filename)
        return data, info, parse_roi(target_roi, info['size'])

    @router.get('/health/live')
    def live():
        return {'live': True}

    @router.get('/health/ready')
    async def ready():
        state = await service.readiness(fresh=True)
        return JSONResponse({'ready': state['ready']}, status_code=200 if state['ready'] else 503)

    @router.get('/api/status')
    async def status():
        return await service.status()

    @router.get('/api/reference/{name}')
    def reference(name: str):
        data = service.assets.thumbnail(name[:-4]) if name.endswith('.jpg') else None
        if data is None:
            return JSONResponse({'error': 'not_found'}, status_code=404)
        return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=3600'})

    @router.post('/api/recognize')
    async def recognize(image: UploadFile = File(...), target_roi: str | None = Form(None)):
        try:
            return {'view': await service.recognize(*await upload(image, target_roi))}
        except Rejected as error:
            return failure(error)

    @router.post('/v1/eval/predict')
    async def predict(image: UploadFile = File(...), target_roi: str | None = Form(None)):
        try:
            return await service.predict(*await upload(image, target_roi))
        except Rejected as error:
            return failure(error)

    return router
