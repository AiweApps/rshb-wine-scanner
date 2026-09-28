"""Immutable inputs of the stage5 evaluator; every read goes through a byte check."""
from pathlib import Path

from rshb_vine.io import read_json, sha256, verify

ROOT = Path(__file__).resolve().parents[2]

PROFILE = 'config/recognition-target-contract-v2-dispatch4.json'
PROFILE_SHA256 = '3980f17fefa622c7dff6a5ba034786a4f81dca59b2b1ac1e093f82ebb3a309ee'
PROFILE_CHECKSUM = '6b8f168147f142b980ac17cd13ff481d4c336abf981257f8f1ac1d1716d8bd38'
PARENT_ENCODER = 'runs/domain-training-v1/B3-round-evaluation/B3-provisional/encoder'
PARENT_EID = '6a5603ba04238a53168eac62212f1a880ae6f8eeb3d94a5acf7f4846be257c07'
PREPROCESSING = 'letterbox384-existing-Encoder-v1'

FILES = {
    'references': ('runs/b3-v2-evaluation-final/references.json',
                   'c830a221a8d94cc3a97e29935c051a404e31a1c43050983b9a9a4d3a196ae837'),
    'current_gallery': ('data/catalog-additions-20260921/B3/gallery.json',
                        '23f68bb24b04147ad56563848d3a0e1edb19e2a308f223a7de18637d03b913cc'),
    'current_vectors': ('data/catalog-additions-20260921/B3/vectors.npy',
                        'fa5348375693f28e947b042259064ab31d6a7f071ffe1832fa6415ac0aad4eca'),
    'white_gallery': ('runs/catalog-additions-alpha-repair-v1/B3/gallery.json',
                      '2d5721df29f3f1bae717fbac5a15f126e7ec277e8c4eff96665043996b49acbd'),
    'white_vectors': ('runs/catalog-additions-alpha-repair-v1/B3/vectors.npy',
                      'd5949dda2ba38c0124c5de70fba15a93b28c025e41356122b27b2dd3921e8782'),
    'signed_white_parent_vectors': ('runs/b3-v2-evaluation-final/fit-v1-eval/parent-gallery-vectors.npy',
                                    '9f29585786325f7f2ea18473045b9d8eb35d04db305c43836488a24b8f7c2a4f'),
    'registry03': ('data/product-identity-v1/snapshot-03-candidate/registry.json',
                   '4364bc0866d1c3939ad05a19d54aed6e52bd4851ad74910cb00180825c1e0230'),
    'validation': ('data/evaluation/web232-v1/validation.json',
                   '9338c8159a8bee55279221877822c3592abbb378a9500163f1fa511048b1c786'),
    'train': ('data/evaluation/web232-v1/train.json',
              'f1d7b4218445344b87399035d9fcf57501c59a0302ffaa77785d8b9e1c2aaa0d'),
    'trainer_contract': ('runs/catalog-training-data-v2/stage5/consumer/trainer-contract.json',
                         '33f593ac280c3cfb1fa71295c021e3e87f73f419e09aa0e17c320ce723e8149f'),
    'trainer': ('rshb_vine/catalog_training_v2/trainer.py',
                '96f0c93f3750505d9cc7e6098719d0748530184d6831b1911ba23cea18322c6d'),
    'probe_admission': ('runs/catalog-training-data-v2/stage5/root-evaluator-probe-admission-v2.json',
                        '7309c0a7cc6b5b0463deca5d8ad26d4cc3435eb6936d732c3d8fcb01487cf662'),
}
CODE = ('rshb_vine/models.py', 'rshb_vine/preprocessing/__init__.py', 'rshb_vine/visual_core.py',
        'rshb_vine/encoder_artifact.py', 'rshb_vine/request_visual_cache.py', 'rshb_vine/recognition_runtime.py',
        'rshb_vine/recognition_factory.py', 'rshb_vine/catalog_training_evaluation_v2/__init__.py',
        'rshb_vine/catalog_training_evaluation_v2/pins.py', 'rshb_vine/catalog_training_evaluation_v2/reindex.py',
        'rshb_vine/catalog_training_evaluation_v2/adapter.py', 'rshb_vine/catalog_training_evaluation_v2/score.py',
        'scripts/catalog_training_evaluate_v2.py')
RECOVERED_IMAGES = 'data/evaluation-source-recovery-v1/images'
SELECTABLE_STEPS = (2895, 4343, 5790)
DIAGNOSTIC_STEPS = (1448,)
OUTPUT_ROOTS = ('runs/catalog-training-data-v2/stage5/evaluator', 'artifacts/catalog-training-v2/evaluator')


def path(name):
    return ROOT / FILES[name][0]


def checked(name):
    target = path(name)
    if sha256(target) != FILES[name][1]:
        raise ValueError('Pinned evaluator input changed: ' + FILES[name][0])
    return target


def read(name, sealed=False):
    value = read_json(checked(name))
    return verify(value) if sealed else value


def output_dir(raw):
    out = Path(raw)
    out = (out if out.is_absolute() else ROOT / out).resolve()
    if not any(out.is_relative_to(ROOT / r) for r in OUTPUT_ROOTS):
        raise ValueError('Evaluator output must stay under ' + ' or '.join(OUTPUT_ROOTS))
    if out.exists():
        raise ValueError('Evaluator output exists; immutable result preserved')
    return out


def code_sha():
    return {rel: sha256(ROOT / rel) for rel in CODE}


def checkpoint_steps():
    contract = read('trainer_contract')
    if (tuple(contract['selection_eligible_checkpoint_steps']) != SELECTABLE_STEPS
            or tuple(contract['diagnostic_only_checkpoint_steps']) != DIAGNOSTIC_STEPS
            or contract['checkpoint_steps'] != sorted(SELECTABLE_STEPS + DIAGNOSTIC_STEPS)):
        raise ValueError('Frozen checkpoint schedule differs from the trainer contract')
    return {'selectable': list(SELECTABLE_STEPS), 'diagnostic_only': list(DIAGNOSTIC_STEPS)}


def product03():
    cards = read('registry03')['cards']
    mapping = {}
    for card in cards:
        if mapping.setdefault(card['slug'], card['product_id']) != card['product_id']:
            raise ValueError('One registry03 slug has two Product IDs: ' + card['slug'])
    return mapping


def recovered_image(sha):
    matches = sorted((ROOT / RECOVERED_IMAGES).glob(sha + '.*'))
    if len(matches) != 1 or sha256(matches[0]) != sha:
        raise ValueError('Recovered image missing or SHA differs: ' + sha)
    return matches[0]
