"""Owner facade of N for rshb_vine.release_next_v1.recipe; the integration imports only this module."""
from rshb_vine.existing_target_parent_v1 import runtime as R

KIND = R.DESCRIPTOR_KIND
SOURCES = R.SOURCES


def load_descriptor(root, path):
    return R.load_descriptor(root, path)


def install(graph, root, descriptor):
    """(state, record) once per loaded graph; refuses a descriptor other than the live one of these sources."""
    if descriptor.get('kind') != KIND or (descriptor.get('parent_release') or {}).get('checksum') != R.PARENT_CHECKSUM:
        raise ValueError('Descriptor is not an existing-target parent component over bde4fa52')
    if R.sources_sha(root) != descriptor['sources_sha256']:
        raise ValueError('Component sources changed since freeze')
    stage, record = R.install(graph)
    return stage, dict(record, descriptor_checksum=descriptor['checksum'])


def run(state, recognize, data, roi=None, bottles='addressed', capture=False):
    return R.run(state, recognize, data, roi, bottles, capture)


def trace(scope):
    return R.trace(scope)
