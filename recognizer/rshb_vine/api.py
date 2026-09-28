"""Thin HTTP transport over the exact same Pipeline as replay."""
import json
import threading
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


def create_app(pipeline):
    app = FastAPI(title='Local RSHB recognition')
    lock = threading.Lock()

    def run(image, roi=None):
        data = image.file.read(20*1024*1024+1)
        target = None
        if roi is not None:
            try:
                target = json.loads(roi)
                if not isinstance(target,list):
                    raise ValueError('ROI must be an array')
            except (ValueError,TypeError) as exc:
                raise HTTPException(422,str(exc))
        # Model calls execute in FastAPI's sync worker; only one request owns adapters.
        if not lock.acquire(blocking=False):
            raise HTTPException(503,'Inference busy; retry later')
        try:
            result = pipeline.recognize(data,target)
        except Exception as exc:
            raise HTTPException(503,'Inference failed: '+type(exc).__name__) from exc
        finally:
            lock.release()
        if result['decision']=='invalid_image':
            raise HTTPException(422,result['reasons'])
        return result

    @app.get('/health/ready')
    def ready():
        return {'ready':True,'snapshot':pipeline.manifest['checksum'],
                'runtime_descriptor_checksum':pipeline.manifest.get('runtime_descriptor_checksum')}

    @app.post('/v1/recognize')
    def recognize(image: UploadFile = File(...), target_roi: str | None = Form(None)):
        return run(image,target_roi)

    @app.post('/v1/eval/predict')
    def predict(image: UploadFile = File(...)):
        result = run(image)
        if result['best_candidate'] is None:
            raise HTTPException(422,'No candidate; refusing a fabricated slug')
        return {'slug':result['best_candidate']}

    return app
