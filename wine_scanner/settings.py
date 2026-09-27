"""Gateway settings from WINE_SCANNER_* environment variables."""
from dataclasses import dataclass
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

BACKEND_HOSTS = ('127.0.0.1', 'localhost', '::1', 'host.docker.internal')
CHECKSUM = re.compile(r'^[0-9a-f]{64}$')


def backend_url(value):
    parts = urlsplit(value)
    if (parts.scheme != 'http' or parts.hostname not in BACKEND_HOSTS or parts.path not in ('', '/')
            or parts.query or parts.fragment or parts.username or parts.password):
        raise ValueError('WINE_SCANNER_BACKEND must be http://<loopback or host.docker.internal>:<port>')
    return f'http://{parts.netloc}'


def checksum(value, name):
    if not CHECKSUM.match(value or ''):
        raise ValueError(f'{name} must be a lowercase 64-hex SHA-256')
    return value


@dataclass(frozen=True)
class Settings:
    assets_dir: Path
    assets_manifest_sha256: str
    expected_profile: str
    backend: str = 'http://127.0.0.1:8175'
    host: str = '127.0.0.1'
    port: int = 8287
    timeout_s: float = 180.0

    @classmethod
    def from_env(cls, env=os.environ):
        def required(name):
            value = env.get(name, '').strip()
            if not value:
                raise ValueError(f'{name} is required')
            return value

        port = int(env.get('WINE_SCANNER_PORT', '8287'))
        if not 1 <= port <= 65535:
            raise ValueError('WINE_SCANNER_PORT must be between 1 and 65535')
        timeout = float(env.get('WINE_SCANNER_TIMEOUT_S', '180'))
        if not 5 <= timeout <= 600:
            raise ValueError('WINE_SCANNER_TIMEOUT_S must be between 5 and 600')
        return cls(assets_dir=Path(required('WINE_SCANNER_ASSETS_DIR')),
                   assets_manifest_sha256=checksum(required('WINE_SCANNER_ASSETS_SHA256'),
                                                   'WINE_SCANNER_ASSETS_SHA256'),
                   expected_profile=checksum(required('WINE_SCANNER_EXPECTED_PROFILE'),
                                             'WINE_SCANNER_EXPECTED_PROFILE'),
                   backend=backend_url(env.get('WINE_SCANNER_BACKEND', cls.backend)),
                   host=env.get('WINE_SCANNER_HOST', cls.host), port=port, timeout_s=timeout)
