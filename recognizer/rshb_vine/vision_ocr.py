"""Persistent, bounded local Vision worker implementing the existing OCR interface."""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time
import uuid
from rshb_vine.io import sha256


class VisionOCR:
    def __init__(self, executable):
        self.executable = Path(executable).resolve()
        self.executable_sha256 = sha256(self.executable)
        self.info = json.loads(subprocess.check_output(
            [str(self.executable), '--info'], timeout=15, text=True))
        if self.info['revision'] != 3 or not {'ru-RU', 'en-US'} <= set(self.info['supported_languages']):
            raise ValueError('Unsupported Vision OCR configuration')
        self.lock = threading.Lock()
        self.worker = self._start_worker()
        atexit.register(self.close)

    def _start_worker(self):
        if sha256(self.executable) != self.executable_sha256:
            raise ValueError('Vision worker binary changed')
        info = json.loads(subprocess.check_output(
            [str(self.executable), '--info'], timeout=15, text=True))
        if info != self.info:
            raise ValueError('Vision OS or reader configuration changed')
        return subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=0)

    def close(self):
        if self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=3)

    def _response_line(self):
        deadline = time.monotonic() + 30
        data = bytearray()
        while b'\n' not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.worker.stdout], [], [], remaining)[0]:
                self.close()
                raise TimeoutError('Vision OCR exceeded 30 seconds')
            chunk = os.read(self.worker.stdout.fileno(), 65536)
            if not chunk:
                self.close()
                raise RuntimeError('Vision worker closed output before completing a response')
            data.extend(chunk)
            if len(data) > 1024*1024:
                self.close()
                raise RuntimeError('Vision response exceeds one MiB')
        line, extra = bytes(data).split(b'\n', 1)
        if extra.strip():
            self.close()
            raise RuntimeError('Unexpected extra Vision response')
        return json.loads(line)

    def read(self, image):
        with self.lock, tempfile.TemporaryDirectory(prefix='rshb-vision-') as directory:
            if self.worker.poll() is not None:
                self.worker = self._start_worker()
            path = Path(directory)/'crop.png'
            image.save(path, format='PNG')
            token = uuid.uuid4().hex
            self.worker.stdin.write((json.dumps({'path': str(path), 'id': token})+'\n').encode())
            self.worker.stdin.flush()
            response = self._response_line()
            if response.get('id') != token or response.get('status') != 'ok' or response.get('revision') != 3:
                raise RuntimeError('Vision OCR failed or response identity mismatched')
            width, height = image.size
            rows = []
            for observation in response['observations']:
                x1, y1, x2, y2 = observation['bbox_bottom_left_normalized']
                left, right = x1*width, x2*width
                top, bottom = (1-y2)*height, (1-y1)*height
                rows.append({'raw_text': observation['raw_text'], 'score': observation['score'],
                             'polygon': [[left, top], [right, top], [right, bottom], [left, bottom]]})
            return rows
