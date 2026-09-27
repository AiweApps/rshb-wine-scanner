"""HTTP client of the running recognition API; only /health/ready and /v1/recognize are used."""
import asyncio
import json

import httpx

from wine_scanner.contracts import Rejected

BUSY_RETRY_S = 2.0
HEALTH_TIMEOUT_S = 4.0


class RecognitionBackend:
    def __init__(self, base_url, timeout_s):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=3.0),
                                        trust_env=False)

    async def close(self):
        await self.client.aclose()

    async def health(self):
        """(ready, runtime_descriptor_checksum or None); a reply without ready: true counts as not ready."""
        try:
            response = await self.client.get('/health/ready', timeout=HEALTH_TIMEOUT_S)
            if response.status_code != 200:
                return False, None
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return False, None
        if not isinstance(body, dict) or body.get('ready') is not True:
            return False, None
        descriptor = body.get('runtime_descriptor_checksum')
        return True, descriptor if isinstance(descriptor, str) else None

    async def recognize(self, data, info, roi):
        form = {'target_roi': json.dumps(roi)} if roi is not None else {}
        files = {'image': ('upload.' + info['format'].lower(), data, info['mime'])}
        for attempt in (1, 2):
            try:
                response = await self.client.post('/v1/recognize', data=form, files=files)
            except httpx.TimeoutException:
                raise Rejected(504, 'timeout', 'Распознавание не уложилось во время ожидания; повторите')
            except httpx.HTTPError:
                raise Rejected(502, 'backend_unreachable', 'Нет связи с сервисом распознавания', retry_after=10)
            if response.status_code == 200:
                try:
                    result = response.json()
                except ValueError:
                    result = None
                if not isinstance(result, dict):
                    raise Rejected(502, 'backend_error', 'Сервис распознавания вернул некорректный ответ')
                return result
            busy = response.status_code == 503 and 'busy' in response.text.lower()
            if busy and attempt == 1:
                await asyncio.sleep(BUSY_RETRY_S)
                continue
            if busy:
                raise Rejected(503, 'busy', 'Сервис распознавания занят, повторите чуть позже', retry_after=5)
            if response.status_code == 422:
                raise Rejected(422, 'invalid_image', 'Сервис не смог обработать изображение')
            raise Rejected(502, 'backend_error', f'Ошибка сервиса распознавания ({response.status_code})')
