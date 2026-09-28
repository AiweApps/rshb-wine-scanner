"""Actual image-byte reader for the sealed stage5 schedule; no model loading."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoImageProcessor

from rshb_vine.catalog_training_v2 import prepare as P
from rshb_vine.domain_augmentation import augment
from rshb_vine.io import local_path, sha256
from rshb_vine.preprocessing import checked_box, decode

CACHE = P.OUT / "cache"
REAL_CACHE = P.ROOT / "runs/b3-v2-runpod-v1/fit-v1/cache"
NEGATIVE_CACHE = P.ROOT / "runs/visual-learning-v1/trial/source-hold-v2/negative-cache-v2"
OLD_CANONICAL_CACHE = P.ROOT / "runs/intraline-visual-v1/canonical-mixed-v1/cache"
MAX_CYCLES = 2


def read(path):
    return json.loads(Path(path).read_text())


def _canonical_indices():
    selected = set()
    for step in P.rows(P.OUT / "steps.jsonl"):
        for task in step["canonical"]:
            for key in ("anchor", "positive", "negative"):
                selected.add(int(task[key]["view_id"].rsplit(":", 1)[1]))
    return sorted(selected)


def prepare_cache():
    """Create only missing stage5 crops from SHA-pinned local originals.

    Existing stage4 source admission is required before this is called by the
    CLI. Reuse an exact old cached view only after source/index/byte checks.
    """
    P.validate_seal()
    refs = read(P.REFS)
    needed = _canonical_indices()
    old_index_path = OLD_CANONICAL_CACHE / "index.json"
    old = {x["reference_index"]: x for x in read(old_index_path)} if old_index_path.exists() else {}
    CACHE.mkdir(parents=True, exist_ok=True)
    listing = []
    last_source, image = None, None
    for index in sorted(needed, key=lambda i: (refs[i]["image_sha256"], i)):
        ref = refs[index]
        source = ref["image_sha256"]
        dest = CACHE / f"v{index:05d}.png"
        method = "exact_bbox_decode_pad_white_384"
        if index in old:
            row = old[index]
            old_path = OLD_CANONICAL_CACHE / row["file"]
            if row["image_sha256"] != source or row["slug"] != ref["slug"] or \
                    row["kind"] != ref["kind"] or sha256(old_path) != row["crop_sha256"]:
                raise ValueError(f"Old canonical crop mismatch: {index}")
            if not dest.exists():
                os.link(old_path, dest)
            if sha256(dest) != row["crop_sha256"]:
                raise ValueError(f"Hardlinked old crop changed: {index}")
            method = "hardlink_verified_old_canonical"
        elif not dest.exists():
            if source != last_source:
                path = local_path(P.ROOT, ref["source_path"])
                if not path.is_file() or sha256(path) != source:
                    raise ValueError(f"Canonical source bytes mismatch: {source}")
                image, _ = decode(path.read_bytes(), max_bytes=1 << 30, max_pixels=36_000_000)
                last_source = source
            if list(image.size) != ref["original_size"]:
                raise ValueError(f"Canonical EXIF geometry mismatch: {index}")
            crop = image.crop(checked_box(ref["bbox"], image.size))
            ImageOps.pad(crop.convert("RGB"), (384, 384), color="white").save(dest)
        with Image.open(dest) as check:
            if check.size != (384, 384) or check.mode not in ("RGB", "RGBA"):
                raise ValueError(f"Canonical cached image has unexpected shape: {index}")
        listing.append({"reference_index": index, "source_sha256": source, "slug": ref["slug"],
                        "kind": ref["kind"], "bbox": ref["bbox"], "file": dest.name,
                        "crop_sha256": sha256(dest), "method": method})
    listing.sort(key=lambda x: x["reference_index"])
    P.write_once(CACHE / "index.json", P.canonical_bytes(listing))
    receipt = {"schema_version":"catalog-training-v2-stage5-cache-v1", "status":"cpu_cache_ready",
               "stage5_manifest_sha256":sha256(P.OUT/"manifest.json"),
               "reference_manifest_sha256":sha256(P.REFS), "index_sha256":sha256(CACHE/"index.json"),
               "n_views":len(listing), "n_hardlinked_old":sum(x["method"].startswith("hardlink") for x in listing)}
    P.write_once(CACHE / "receipt.json", P.canonical_bytes(receipt))
    return receipt


def validate_cache(*, all_bytes=False):
    receipt, listing = read(CACHE/"receipt.json"), read(CACHE/"index.json")
    if receipt["stage5_manifest_sha256"] != sha256(P.OUT/"manifest.json") or \
            receipt["reference_manifest_sha256"] != sha256(P.REFS) or \
            receipt["index_sha256"] != sha256(CACHE/"index.json") or receipt["n_views"] != len(listing):
        raise ValueError("Stage5 cache seal drift")
    if all_bytes:
        for item in listing:
            if sha256(CACHE/item["file"]) != item["crop_sha256"]:
                raise ValueError("Canonical cache bytes drift: "+item["file"])
    return {x["reference_index"]:x for x in listing}


class PreparedSteps:
    def __init__(self, *, portable=False):
        self.manifest = P.validate_runtime_seal() if portable else P.validate_seal()
        if self.manifest.get("fit_admitted") is not False or self.manifest.get("schema_version") != "catalog-training-v2-stage5-consumer-v1":
            raise ValueError("Invalid stage5 consumer manifest")
        self.steps = list(P.rows(P.OUT/"steps.jsonl"))
        if len(self.steps) != self.manifest["steps"] or sha256(P.OUT/"steps.jsonl") != self.manifest["output_pins"]["steps.jsonl"]:
            raise ValueError("Stage5 schedule seal drift")
        self.rows = {x["id"]:x for x in read(P.REAL)["rows"]}
        self.cache = validate_cache()
        self.negative_index = {x["reference_index"]:x for x in read(NEGATIVE_CACHE/"negative-cache-manifest.json")}
        self.parent = P.ROOT / self.manifest["parent_encoder"]
        self.processor = None

    def __len__(self):
        return len(self.steps)

    @staticmethod
    def _photometric(image, source, presentation):
        digest = hashlib.sha256(f"catalog-training-v2:{source}:{presentation}".encode()).digest()
        brightness = 0.94 + digest[0]/255*0.12
        contrast = 0.94 + digest[1]/255*0.12
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(brightness)).enhance(contrast)

    def _real(self, row_id, *, augmentation_index=None):
        row = self.rows[row_id]
        path = REAL_CACHE/f"{row_id}.png"
        if sha256(path) != row["cached_crop_sha256"]:
            raise ValueError("Real crop bytes drift: "+row_id)
        with Image.open(path) as fp:
            image = fp.convert("RGB")
            if augmentation_index is not None:
                image, _ = augment(image, row["source_sha256"], augmentation_index,
                                   self.manifest["real_augmentation_seed"])
            return ImageOps.pad(image, (384,384), color="white")

    def _mined(self, index):
        row = self.negative_index[index]
        path = NEGATIVE_CACHE/row["file"]
        if sha256(path) != row["crop_sha256"]:
            raise ValueError("Old mined crop bytes drift: "+str(index))
        with Image.open(path) as fp:
            return ImageOps.pad(fp.convert("RGB"), (384,384), color="white")

    def _canonical(self, view, *, presentation=None):
        source_id, kind, index_text = view["view_id"].rsplit(":",2)
        index = int(index_text)
        row = self.cache[index]
        if row["source_sha256"] != view["source_sha256"] or row["source_sha256"] != source_id or \
                row["kind"] != kind or ("bbox" in view and row["bbox"] != view["bbox"]):
            raise ValueError("Canonical schedule/cache view drift")
        path = CACHE/row["file"]
        if sha256(path) != row["crop_sha256"]:
            raise ValueError("Canonical crop bytes drift: "+row["file"])
        with Image.open(path) as fp:
            image = fp.convert("RGB")
            if presentation is not None:
                image = self._photometric(image, view["source_sha256"], presentation)
            return image.copy()

    def __getitem__(self, global_step):
        # A later cycle replays the sealed step file; only real-query augmentation advances.
        cycle, step_index = divmod(global_step, len(self.steps))
        if global_step < 0 or cycle >= MAX_CYCLES:
            raise ValueError("Global step outside the sealed cycle bound")
        step = self.steps[step_index]
        if step["step"] != step_index:
            raise ValueError("Stage5 step index drift")
        real = step["real"]
        canon = step["canonical"]
        images = []
        images += [self._real(x["query_id"], augmentation_index=global_step*4+x["old_position"]) for x in real]
        images += [self._canonical(x["anchor"], presentation=x["presentation_index"]) for x in canon]
        images += [self._real(x["reference_id"]) for x in real]
        images += [self._canonical(x["positive"]) for x in canon]
        images += [self._mined(m["reference_index"]) for x in real for m in x["mined"]]
        images += [self._canonical(x["negative"]) for x in canon]
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.parent, local_files_only=True, use_fast=False)
        tensor = self.processor(images=images, return_tensors="pt")["pixel_values"]
        if tensor.shape != (len(images),3,384,384) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Processor tensor shape/nonfinite error")
        return {"pixels":tensor, "step":step, "global_step":global_step, "cycle":cycle, "real_count":len(real), "canonical_count":len(canon),
                "mined_count":sum(len(x["mined"]) for x in real),
                "layout":"real_q,canonical_q,real_ref,canonical_positive,real_mined,canonical_negative"}
