"""Exact ordered RGB-batch reuse inside one recognition request only.

The original encoder owns preprocessing, features, normalization and devices.
This delegate only memoizes complete encode calls; it never rebatches images.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import time

import numpy as np
from PIL import Image
from rshb_vine.io import digest


class _RequestState:
    def __init__(self, enabled, capture_retrieval=False):
        self.enabled = enabled
        self.batches = {}
        self.encoders = {}
        self.capture_retrieval = enabled and capture_retrieval
        self.vector_inputs = {}
        self.retrievals = {}
        self.retrieval_counts = {'recorded': 0, 'ambiguous': 0, 'reused': 0, 'missed': 0}

    def counters(self, encoder_id):
        return self.encoders.setdefault(encoder_id, {
            'calls': 0, 'images_requested': 0, 'hit_batches': 0, 'hit_images': 0,
            'miss_batches': 0, 'bypass_batches': 0, 'actual_encode_batches': 0,
            'actual_encode_images': 0, 'key_seconds': 0., 'encode_seconds': 0.,
        })

    def snapshot(self):
        return {'enabled': self.enabled, 'scope': 'one_top_level_recognize_call',
                'key': 'encoder_id_and_complete_ordered_RGB_pixel_batch_mode_size_sha256',
                'per_image_rebatching': False, 'features_changed': False,
                'retained_batches': len(self.batches), 'encoders': deepcopy(self.encoders),
                'recorded_raw_retrieval': dict(self.retrieval_counts)}

    def bind_vectors(self, encoder_id, pixels, vectors):
        if self.capture_retrieval:
            for pixel, row in zip(pixels, vectors, strict=True):
                key = (encoder_id, sha256(row.tobytes()).digest())
                self.vector_inputs.setdefault(key, set()).add(pixel)


class RequestVisualCache:
    def __init__(self, *, capture_retrieval=False):
        self._state = ContextVar('recognition_visual_request_' + str(id(self)), default=None)
        self._owners = {}
        self._capture_retrieval = capture_retrieval

    @contextmanager
    def request(self, *, enabled):
        if type(enabled) is not bool:
            raise TypeError('Memo mode must be a boolean')
        state = _RequestState(enabled, self._capture_retrieval)
        token = self._state.set(state)
        try:
            yield state
        finally:
            state.batches.clear()
            state.vector_inputs.clear()
            state.retrievals.clear()
            self._state.reset(token)

    def wrap(self, encoder, encoder_id):
        if not isinstance(encoder_id, str) or not encoder_id:
            raise ValueError('Memo requires a verified encoder ID')
        previous = self._owners.setdefault(encoder_id, encoder)
        if previous is not encoder:
            raise ValueError('One memo encoder ID cannot refer to different live models')
        return _EncoderDelegate(encoder, encoder_id, self)

    def wrap_index(self, index, encoder_id):
        if index.encoder_id != encoder_id:
            raise ValueError('Recorded index has another encoder')
        return _IndexDelegate(index, encoder_id, self)


def _pixels(batch):
    return tuple((im.mode, im.size, sha256(im.tobytes()).digest()) for im in batch)


class _EncoderDelegate:
    def __init__(self, encoder, encoder_id, cache):
        self._encoder = encoder
        self._encoder_id = encoder_id
        self._cache = cache

    def __getattr__(self, name):
        # In particular, features remains the original bound differentiable
        # method. Neither object-gate vectors nor training tensors are replaced.
        return getattr(self._encoder, name)

    def encode(self, images):
        state = self._cache._state.get()
        if state is None:
            return self._encoder.encode(images)
        batch = list(images)
        counts = state.counters(self._encoder_id)
        counts['calls'] += 1
        counts['images_requested'] += len(batch)
        key = None
        if state.enabled and batch and all(isinstance(im, Image.Image) and im.mode == 'RGB' for im in batch):
            started = time.perf_counter()
            key = (self._encoder_id, _pixels(batch))
            counts['key_seconds'] += time.perf_counter() - started
            if key in state.batches:
                counts['hit_batches'] += 1
                counts['hit_images'] += len(batch)
                return state.batches[key].copy()
            counts['miss_batches'] += 1
        else:
            # RGB decode is the runtime contract. Other modes can have palette
            # or transparency metadata absent from pixel bytes: do not memoize.
            counts['bypass_batches'] += 1
        started = time.perf_counter()
        values = self._encoder.encode(batch)
        counts['encode_seconds'] += time.perf_counter() - started
        counts['actual_encode_batches'] += 1
        counts['actual_encode_images'] += len(batch)
        if key is not None:
            if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 2 or values.shape[0] != len(batch):
                raise ValueError('Original encoder returned an unexpected batch contract')
            # Store a separate array. The first caller also retains its own
            # original result; its downstream mutations cannot poison the memo.
            state.batches[key] = values.copy()
            state.bind_vectors(self._encoder_id, key[1], values)
        return values


class _IndexDelegate:
    """Observe original search calls; never assemble or change encoder batches.

    A reused result must already have been computed by the frozen pipeline for
    the exact ordered crop pixels AND complete view descriptors. Vector hashes
    only bind that prior search to encoder inputs; they are never new embeddings.
    Ambiguous input bindings or differing recorded results force a normal search.
    """
    def __init__(self, index, encoder_id, cache):
        self._index, self._encoder_id, self._cache = index, encoder_id, cache

    def __getattr__(self, name):
        return getattr(self._index, name)

    def search(self, query, views, encoder_id):
        state = self._cache._state.get()
        key = None
        if state is not None and state.capture_retrieval and encoder_id == self._encoder_id:
            inputs = [state.vector_inputs.get((encoder_id, sha256(row.tobytes()).digest()), set())
                      for row in np.asarray(query, dtype='float32')]
            if len(inputs) == len(views) and inputs and all(len(x) == 1 for x in inputs):
                key = (encoder_id, tuple(next(iter(x)) for x in inputs), digest([asdict(v) for v in views]))
        result = self._index.search(query, views, encoder_id)
        if key is not None:
            signature = digest(result)
            if key not in state.retrievals:
                state.retrievals[key] = (signature, deepcopy(result))
                state.retrieval_counts['recorded'] += 1
            elif state.retrievals[key] is not None and state.retrievals[key][0] != signature:
                state.retrievals[key] = None
                state.retrieval_counts['ambiguous'] += 1
        return result

    def reuse(self, images, views, encoder_id):
        state = self._cache._state.get()
        if state is None or not state.capture_retrieval:
            return None
        if encoder_id != self._encoder_id:
            raise ValueError('Cannot reuse another encoder space')
        key = (encoder_id, _pixels(images), digest([asdict(v) for v in views]))
        cached = state.retrievals.get(key)
        if cached is None:
            state.retrieval_counts['missed'] += 1
            return None
        state.retrieval_counts['reused'] += 1
        return deepcopy(cached[1])
