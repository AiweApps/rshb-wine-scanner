"""CPU preparation/verification of stage5 plus the receipt-guarded trainer entry points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.catalog_training_v2 import prepare as P


def describe(portable=False):
    m = P.validate_runtime_seal() if portable else P.validate_seal()
    return {"status":m["status"],"fit_admitted":m["fit_admitted"],
            "steps":m["steps"],"canonical_per_batch_proposal":m["canonical_per_batch_proposal"],
            "real_per_batch_max":m["real_per_batch_max"],"loss_weights":m["loss_weights"],
            "parent_encoder":m["parent_encoder"],"input_pins":m["input_pins"],
            "output_pins":m["output_pins"]}


def preflight(portable=False):
    from rshb_vine.catalog_training_v2.data import validate_cache, REAL_CACHE, NEGATIVE_CACHE
    from rshb_vine.io import sha256
    m = P.validate_runtime_seal() if portable else P.validate_seal()
    if m["fit_admitted"] is not False or m["schema_version"] != "catalog-training-v2-stage5-consumer-v1":
        raise ValueError("Invalid stage5 fit status/schema")
    for rel,digest in m["code_pins"].items():
        if sha256(P.ROOT/rel) != digest:
            raise ValueError("Stage5 code changed: "+rel)
    for name,digest in m["output_pins"].items():
        if sha256(P.OUT/name) != digest:
            raise ValueError("Stage5 output changed: "+name)
    for name,digest in m["parent_files_sha256"].items():
        if sha256(P.ROOT/m["parent_encoder"]/name) != digest:
            raise ValueError("Parent checkpoint changed: "+name)
    cache = validate_cache(all_bytes=True)
    real = {x["id"]:x for x in P.read(P.REAL)["rows"]}
    mined_listing = {x["reference_index"]:x for x in P.read(NEGATIVE_CACHE/"negative-cache-manifest.json")}
    real_ids, mined_ids = set(), set()
    for step in P.rows(P.OUT/"steps.jsonl"):
        for item in step["real"]:
            real_ids.update((item["query_id"],item["reference_id"]))
            mined_ids.update(m["reference_index"] for m in item["mined"])
    for row_id in sorted(real_ids):
        if sha256(REAL_CACHE/f"{row_id}.png") != real[row_id]["cached_crop_sha256"]:
            raise ValueError("Real cached crop changed: "+row_id)
    for index in sorted(mined_ids):
        row = mined_listing[index]
        if sha256(NEGATIVE_CACHE/row["file"]) != row["crop_sha256"]:
            raise ValueError("Mined cached crop changed: "+str(index))
    return {"status":"cpu_stage5_preflight_passed","portable":portable,"fit_admitted":False,
            "real_crops":len(real_ids),"mined_crops":len(mined_ids),"canonical_crops":len(cache),
            "parent_files_checked":len(m["parent_files_sha256"])}


def data_probe(indices, portable=False):
    from rshb_vine.catalog_training_v2.data import PreparedSteps
    d = PreparedSteps(portable=portable)
    observations = []
    for i in indices:
        b = d[i]
        n,m,k = b["real_count"],b["canonical_count"],b["mined_count"]
        step = b["step"]
        if b["pixels"].shape[0] != 2*n+3*m+k or len(step["real_excluded_base"]) != n:
            raise ValueError("Probe tensor/mask shape mismatch")
        if any(len(row)!=n or row[j] is not False for j,row in enumerate(step["real_excluded_base"])):
            raise ValueError("Probe real positive/mask mismatch")
        if any(any(col<0 or col>=n+k for col in columns) for columns in step["real_local_columns"]):
            raise ValueError("Probe real local column mismatch")
        observations.append({"step":i,"tensor_shape":list(b["pixels"].shape),"real":n,"canonical":m,"mined":k,
                             "real_allowed_offdiag":sum(not excluded for j,row in enumerate(step["real_excluded_base"])
                                                         for z,excluded in enumerate(row) if j!=z),
                             "canonical_product_weights":[x["product_weight"] for x in step["canonical"]]})
    return {"status":"cpu_actual_pixels_probe_passed","model_inference":False,"optimizer_started":False,
            "observations":observations}


def transport_list():
    """Exact portable file allowlist; no upstream stage1-4 graph needed remotely."""
    from rshb_vine.catalog_training_v2.data import CACHE, REAL_CACHE, NEGATIVE_CACHE, validate_cache
    from rshb_vine.io import sha256
    m = P.validate_seal()
    cache = validate_cache(all_bytes=True)
    selected = {P.OUT/"manifest.json",P.RUN/"receipt.json",P.OUT/"steps.jsonl",P.OUT/"real-holds.jsonl",
                P.OUT/"real-query-loss-coverage.jsonl",P.OUT/"coverage.json",
                CACHE/"index.json",CACHE/"receipt.json",P.REAL,P.REFS,
                NEGATIVE_CACHE/"negative-cache-manifest.json"}
    selected.update(P.ROOT/rel for rel in m["code_pins"])
    selected.add(P.ROOT/"rshb_vine/__init__.py")
    selected.update(P.ROOT/m["parent_encoder"]/name for name in m["parent_files_sha256"])
    selected.update(CACHE/row["file"] for row in cache.values())
    real = {x["id"]:x for x in P.read(P.REAL)["rows"]}
    neg = {x["reference_index"]:x for x in P.read(NEGATIVE_CACHE/"negative-cache-manifest.json")}
    for step in P.rows(P.OUT/"steps.jsonl"):
        for item in step["real"]:
            for row_id in (item["query_id"],item["reference_id"]):
                selected.add(REAL_CACHE/f"{row_id}.png")
            for mine in item["mined"]:
                selected.add(NEGATIVE_CACHE/neg[mine["reference_index"]]["file"])
    files = []
    for path in sorted(selected):
        if not path.is_file():
            raise ValueError("Transport input missing: "+str(path))
        files.append({"path":str(path.relative_to(P.ROOT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    out = {"schema_version":"catalog-training-v2-stage5-transport-allowlist-v1", "fit_admitted":False,
           "manifest_sha256":sha256(P.OUT/"manifest.json"),"files":files,
           "total_bytes":sum(x["bytes"] for x in files),"n_files":len(files)}
    P.write_once(P.RUN/"transport-allowlist.json",P.canonical_bytes(out))
    return {"status":"allowlist_ready","path":str((P.RUN/"transport-allowlist.json").relative_to(P.ROOT)),
            "n_files":len(files),"total_bytes":out["total_bytes"],"sha256":sha256(P.RUN/"transport-allowlist.json")}


def trainer_contract():
    from rshb_vine.catalog_training_v2.trainer import expected_recipe
    recipe = expected_recipe(P.validate_runtime_seal())
    P.write_once(P.RUN/"trainer-contract.json",P.canonical_bytes(recipe))
    return {"status":"recipe_contract_not_frozen","path":str((P.RUN/"trainer-contract.json").relative_to(P.ROOT)),
            "sha256":sha256_path(P.RUN/"trainer-contract.json"),
            "note":"root freezes an identical recipe file; this contract grants no fit admission"}


def sha256_path(path):
    from rshb_vine.io import sha256
    return sha256(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=("build","prepare-cache","describe","preflight","data-probe","transport-list",
                                           "trainer-contract","profile","fit","export"))
    parser.add_argument("--portable",action="store_true",help="verify sealed transport copy without stage1-4 rebuild")
    parser.add_argument("--steps",default="0,1,1447,2894",help="at most 12 deterministic actual-pixel batches")
    for name in ("recipe","admission","spend","lease","profile","resume-receipt","output","checkpoint"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--workers",type=int,choices=(0,2,4,8),default=8)
    a = parser.parse_args()
    if a.command in ("profile","fit"):
        from rshb_vine.catalog_training_v2.trainer import run
        if a.recipe is None or a.output is None:
            raise ValueError("profile/fit require --recipe and --output")
        a.mode = a.command
        run(a)
        return
    if a.command == "export":
        from rshb_vine.catalog_training_v2.trainer import export
        if a.checkpoint is None or a.output is None or a.recipe is None:
            raise ValueError("export requires --checkpoint, --recipe and --output")
        out = export(a.checkpoint,a.output,a.recipe)
    elif a.command == "trainer-contract":
        out = trainer_contract()
    elif a.command == "build":
        if a.portable: raise ValueError("Build requires fresh local admission")
        out = P.build()
    elif a.command == "prepare-cache":
        if a.portable: raise ValueError("Cache preparation requires local source audit")
        from rshb_vine.catalog_training_v2.data import prepare_cache
        out = prepare_cache()
    elif a.command == "describe":
        out = describe(a.portable)
    elif a.command == "preflight":
        out = preflight(a.portable)
    elif a.command == "transport-list":
        if a.portable: raise ValueError("Transport list builds from local closure")
        out = transport_list()
    else:
        indices = [int(x) for x in a.steps.split(",")]
        if not indices or len(indices)>12 or len(set(indices))!=len(indices) or any(x<0 for x in indices):
            raise ValueError("Data probe requires 1..12 distinct nonnegative steps")
        out = data_probe(indices,a.portable)
    print(json.dumps(out,ensure_ascii=False,sort_keys=True))


if __name__ == "__main__":
    main()
