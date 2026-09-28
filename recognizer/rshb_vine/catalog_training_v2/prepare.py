"""Merge the admitted canonical cycle with the old real branch, without fitting.

Stage 1-4 are rebuilt and checked once by ``preparation.admission``. The resulting
stage 5 step file is sealed for a portable reader; batch loading never rebuilds
the graph or silently treats a different ProductID as a negative.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rshb_vine.io import sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/catalog-training-data-v2/stage5"
RUN = ROOT / "runs/catalog-training-data-v2/stage5/consumer"
STAGE3 = ROOT / "data/catalog-training-data-v2/stage3"
STAGE4 = ROOT / "data/catalog-training-data-v2/stage4"
REAL = ROOT / "runs/b3-v2-runpod-v1/signature-prep-v1/manifest-for-root-signature.json"
OLD = ROOT / "runs/visual-learning-v1/trial/source-hold-v2/mined-schedule.json"
OLD_RECIPE = ROOT / "runs/visual-learning-v1/trial/source-hold-v2/recipe-fit.json"
REFS = ROOT / "runs/b3-v2-evaluation-final/references.json"
REG01 = ROOT / "data/product-identity-v1/snapshot-01/registry.json"
REG03 = ROOT / "data/product-identity-v1/snapshot-03-candidate/registry.json"
SOURCE_LEDGER = ROOT / "data/catalog-training-data-v2/stage2/source-ledger.jsonl"
MASK = ROOT / "runs/comparison-groups-v1/qa-fit-union-v1/qa-card-revision-v1/frozen-01/negative-mask-candidate-v1.json"
NEGATIVE_MANIFEST = ROOT / "runs/visual-learning-v1/trial/source-hold-v2/negative-cache-v2/negative-cache-manifest.json"
OLD_LEDGER = ROOT / "runs/visual-learning-v1/pair-ledger-v2.jsonl"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Sealed stage5 output differs: {path}")
    if not path.exists():
        path.write_bytes(content)


def canonical_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def jsonl_bytes(items):
    return b"".join(canonical_bytes(x) for x in items)


def _stats(values):
    values = sorted(values)
    return {"min": values[0], "median": (values[(len(values)-1)//2]+values[len(values)//2])/2,
            "max": values[-1], "zero": values.count(0), "n": len(values)} if values else None


def _product_maps():
    slug01 = defaultdict(set)
    for card in read(REG01)["cards"]:
        slug01[card["product_id"]].add(card["slug"])
    slug03 = {card["slug"]: card["product_id"] for card in read(REG03)["cards"]}
    def convert(old):
        found = {slug03[s] for s in slug01[old] if s in slug03}
        if not found:
            raise ValueError("Old real ProductID has no current slug: " + old)
        return sorted(found)
    return convert, slug03


def build(*, canonical_per_batch=4):
    from rshb_vine.catalog_training_data_v2 import preparation
    if canonical_per_batch != 4:
        raise ValueError("Only explicit proposed canonical_per_batch=4 is prepared")
    admission = preparation.admission()  # mandatory fresh check before consuming stage3/4
    if not admission["data_preparation_valid"] or admission["blocks"]:
        raise ValueError("Stage3/4 admission failed: " + repr(admission["blocks"]))
    old, real = read(OLD), read(REAL)
    if old["ledger_sha256"] != sha256(OLD_LEDGER):
        raise ValueError("Old mined negative admission ledger changed")
    mined_ledger = {x["view_id"]:x for x in rows(OLD_LEDGER)}
    refs = read(REFS)
    canonical = list(rows(STAGE4 / "schedule.jsonl"))
    if len(canonical) != 11577 or len(old["steps"]) < (len(canonical)+3)//4:
        raise ValueError("Cycle or old real schedule length changed")
    row_by_id = {x["id"]: x for x in real["rows"]}
    reference_rows_by_source_view = defaultdict(list)
    for row in real["rows"]:
        if row["role"] == "encoder_fit" and row["asset_role"] == "reference":
            reference_rows_by_source_view[(row["source_sha256"],row["view"])].append(row)
    source_rows = {x["original_sha256"]:x for x in rows(SOURCE_LEDGER)}
    source_admission = {source:x["consumers"]["encoder_known_catalog"]["decision"]
                        for source,x in source_rows.items()}
    donors_by_product = defaultdict(set)
    for source, source_row in source_rows.items():
        if source_admission[source] == "admitted_candidate":
            donors_by_product[source_row["product03"]].add(source)
    convert, slug03 = _product_maps()
    eligible, real_semantic_negative, graph_decisions = {}, {}, {}
    for edge in rows(STAGE3 / "edges.jsonl"):
        key = tuple(sorted((edge["product_a03"], edge["product_b03"])))
        graph_decisions[key] = edge["decision"]
        if edge["decision"] in {"negative_metadata_supported", "negative_legacy_pair_qa"} and \
                edge.get("negative_evidence") and \
                edge.get("trainable_exclusion") in (None,"no_admitted_source:a","no_admitted_source:b","no_admitted_source:a,b"):
            real_semantic_negative[key] = edge["negative_evidence"]
        if edge["decision"] in {"negative_metadata_supported", "negative_legacy_pair_qa"} and \
                edge["sources"]["a"]["admitted"] and edge["sources"]["b"]["admitted"]:
            eligible[key] = {"decision": edge["decision"], "relation": edge["producer_relationship"],
                             "evidence": edge["negative_evidence"]}
    ref03 = [slug03[r["slug"]] for r in refs]
    old_safe = {}
    for name,basis in (("ordinary_negative_pairs","prior_ordinary"),("same_producer_hard_negatives","prior_hard")):
        for edge in real[name]:
            if edge.get("safe_negative_admission") is not True or edge.get("encoder_fit_admission") is not True:
                raise ValueError("Prior real edge lacks explicit admission")
            if basis == "prior_ordinary" and edge.get("policy") != "root_accepted_prior_source_identity_different_producer_ordinary_negative":
                raise ValueError("Prior ordinary edge policy drift")
            old_safe[tuple(sorted((edge["product_a"],edge["product_b"])))]=basis
    def edge_basis(a, b, prior_basis=None):
        if not a or not b:
            return "masked_empty_product"
        seen = set()
        for x in a:
            for y in b:
                if x == y:
                    return "masked_same_acceptable_product"
                key = tuple(sorted((x,y)))
                if key in eligible:
                    seen.add("stage3_admitted")
                elif key in real_semantic_negative:
                    # The relation remains evidenced even when this ProductID
                    # lacks a canonical donor. Actual real-query admission and
                    # actual negative-reference source are checked separately.
                    seen.add("stage3_real_role_semantic_negative")
                elif graph_decisions.get(key) not in (None,"excluded_not_evaluated"):
                    return "masked_stage3_"+graph_decisions[key]
                elif prior_basis:
                    seen.add(prior_basis)
                else:
                    return "masked_no_admitted_edge"
        return "+".join(sorted(seen))
    def key_for(a, b):
        return [list(k) for k in sorted({tuple(sorted((x, y))) for x in a for y in b})]
    steps, holds = [], []
    counts = Counter()
    kept_queries, kept_products = set(), set()
    c_products = Counter(t["anchor"]["product03"] for t in canonical)
    c_product_mass = len(canonical) / len(c_products)
    weighted_producers = defaultdict(float)
    weighted_negative_products = defaultdict(float)
    producer_products = defaultdict(set)
    for i in range((len(canonical)+3)//4):
        raw = old["steps"][i]
        real_items = []
        for pos, (p, qid, rid) in enumerate(zip(raw["products"], raw["query_rows"], raw["reference_rows"])):
            q, r = row_by_id[qid], row_by_id[rid]
            reason = None
            if q.get("role") != "encoder_fit" or r.get("role") != "encoder_fit":
                reason = "legacy_role_not_encoder_fit"
            elif source_admission.get(r["source_sha256"]) != "admitted_candidate":
                reason = "canonical_source_" + source_admission.get(r["source_sha256"], "not_in_stage2")
            elif not q.get("encoder_fit_admission") or not r.get("encoder_fit_admission"):
                reason = "legacy_encoder_admission_false"
            replacement = None
            if reason and reason.startswith("canonical_source_"):
                original_source = source_rows[r["source_sha256"]]
                original_product03 = original_source["product03"]
                allowed03 = {slug03[x] for x in q.get("training_identity_evidence",{}).get("acceptable_slugs",[])
                             if x in slug03}
                old_group = q.get("training_identity_status") == "admitted_equivalence" and \
                            original_product03 in allowed03 and \
                            set(q.get("training_group_acceptable_product_ids") or []) == set(q.get("acceptable_product_ids") or [])
                donor_products = {original_product03} | (allowed03 if old_group else set())
                candidates = []
                for product in sorted(donor_products & allowed03):
                    for source in sorted(donors_by_product[product]-{r["source_sha256"],q["source_sha256"]}):
                        for candidate in reference_rows_by_source_view[(source,r["view"])]:
                            candidates.append((source,candidate["id"],product,candidate))
                if candidates:
                    source, donor_id, product03, donor = sorted(candidates,key=lambda x:(x[0],x[1]))[0]
                    if donor["source_sha256"] != source or source_admission[source] != "admitted_candidate":
                        raise ValueError("Replacement donor admission drift")
                    replacement = {"old_reference_id":rid,"new_reference_id":donor_id,
                                   "old_source_sha256":r["source_sha256"],"new_source_sha256":source,
                                   "old_product03":original_product03,"new_product03":product03,
                                   "basis":"exact_product03" if product03==original_product03 else "old_admitted_training_equivalence",
                                   "training_identity_evidence":q["training_identity_evidence"] if old_group else None}
                    r, rid, reason = donor, donor_id, None
                    counts["real_positive_replaced"] += 1
            if reason:
                source_row = source_rows.get(r["source_sha256"])
                original_product03 = source_row["product03"] if source_row else None
                holds.append({"old_step": i, "old_position": pos, "query_id": qid, "reference_id": rid,
                              "query_source_sha256": q["source_sha256"], "reference_source_sha256": r["source_sha256"],
                              "reference_product03": original_product03,
                              "source_held_reasons":source_row["consumers"]["encoder_known_catalog"].get("held_reasons",[]) if source_row else [],
                              "admitted_donor_rows_compatible_view":0, "reason": reason})
                counts["real_held_" + reason] += 1
                continue
            products = sorted(set().union(*(set(convert(x)) for x in q.get("acceptable_product_ids", []))))
            if not products:
                raise ValueError("Real query lacks acceptable current ProductID: "+qid)
            ref_products = {slug03[x["slug"]] for x in r["identity_evidence"] if x["slug"] in slug03}
            if not set(products) & ref_products:
                raise ValueError("Real positive has no current ProductID intersection")
            mines = []
            ledger = mined_ledger.get(qid)
            ledger_negatives = {x["reference_index"]:x for x in ledger["negatives"]} if ledger else {}
            for ref_index in raw["mined_reference_index"][pos]:
                ref = refs[ref_index]
                ledger_entry = ledger_negatives.get(ref_index)
                old_basis = old_safe.get(tuple(sorted((p,ref["product_id"]))))
                if ledger_entry and ledger_entry.get("class") == "negative_catalogue_distinct" and \
                        ledger_entry.get("reference_sha256") == ref["image_sha256"]:
                    old_basis = "prior_mined_ledger"
                basis = edge_basis(products,[ref03[ref_index]],old_basis)
                reason_mined = None
                if source_admission.get(ref["image_sha256"]) != "admitted_candidate":
                    reason_mined = "source_not_stage2_admitted"
                elif basis.startswith("masked_"):
                    reason_mined = basis
                if reason_mined:
                    counts["old_mined_removed_" + reason_mined] += 1
                else:
                    mines.append({"reference_index": ref_index, "product03": ref03[ref_index],
                                  "source_sha256": ref["image_sha256"], "edge_keys03": key_for(products,[ref03[ref_index]]),
                                  "basis":basis,"old_ledger_view_id":qid if "prior_mined_ledger" in basis else None})
                    counts["old_mined_retained"] += 1
                    counts["old_mined_retained_"+basis] += 1
            real_items.append({"old_position": pos, "old_product_id":p, "query_id": qid, "reference_id": rid,
                               "acceptable_products03": products, "query_source_sha256": q["source_sha256"],
                               "reference_source_sha256": r["source_sha256"], "positive_replacement":replacement,
                               "mined": mines})
            kept_queries.add(qid); kept_products.update(products)
        mask, local, inbatch_basis = [], [], []
        for a, left in enumerate(real_items):
            row, local_row, basis_row = [], [], []
            for b, right in enumerate(real_items):
                prior = old_safe.get(tuple(sorted((left["old_product_id"],right["old_product_id"]))))
                basis = "diagonal_positive" if a==b else edge_basis(left["acceptable_products03"],right["acceptable_products03"],prior)
                if a != b and left["query_source_sha256"] == right["reference_source_sha256"]:
                    basis = "masked_same_source"
                allow = a==b or not basis.startswith("masked_")
                row.append(not allow)
                basis_row.append(basis)
                if a != b and allow:
                    local_row.append(b)
                    counts["real_inbatch_allowed_"+basis] += 1
                elif a != b:
                    counts["real_inbatch_"+basis] += 1
            local_row.extend(len(real_items)+k for k, mine in enumerate(
                [m for item in real_items for m in item["mined"]])
                if any(mine is m for m in left["mined"]))
            mask.append(row); local.append(local_row); inbatch_basis.append(basis_row)
        tasks = canonical[i*4:(i+1)*4]
        ct = []
        for task in tasks:
            p = task["anchor"]["product03"]
            weight = c_product_mass / c_products[p]
            weighted_producers[task["producer"]] += weight
            weighted_negative_products[task["negative"]["product03"]] += weight
            producer_products[task["producer"]].add(p)
            ct.append({"presentation_index": task["presentation_index"], "anchor": task["anchor"],
                       "positive": task["positive"], "negative": task["negative"],
                       "producer": task["producer"], "product_weight": weight})
        steps.append({"step": i, "real": real_items, "real_excluded_base": mask,
                      "real_inbatch_basis":inbatch_basis,"real_local_columns": local, "canonical": ct})
    sources = sorted({x["image_sha256"] for x in refs})
    ref_used = sorted({int(v["view_id"].rsplit(":",1)[1]) for task in canonical for v in
                       (task["anchor"],task["positive"],task["negative"] if task["negative"] else {}) if v})
    query_loss = defaultdict(lambda:Counter())
    for step in steps:
        n = len(step["real"])
        for i,item in enumerate(step["real"]):
            c = query_loss[item["query_id"]]
            c["presentations"] += 1
            c["forward_inbatch_negative_cells"] += sum(not step["real_excluded_base"][i][j]
                                                       for j in range(n) if j != i)
            c["reverse_inbatch_negative_cells"] += sum(not step["real_excluded_base"][j][i]
                                                       for j in range(n) if j != i)
            c["forward_mined_negative_cells"] += len(item["mined"])
            c["local_negative_cells"] += len(step["real_local_columns"][i])
    query_loss_rows = [{"query_id":qid,**{k:int(v) for k,v in sorted(c.items())},
                        "effective_negative_loss":sum(v for k,v in c.items() if k!="presentations")>0}
                       for qid,c in sorted(query_loss.items())]
    no_effect = [x["query_id"] for x in query_loss_rows if not x["effective_negative_loss"]]
    coverage = {"real_query_views_retained": len(kept_queries), "real_anchor_products03_retained": len(kept_products),
                "real_query_views_old_cycle": len({q for e in old["steps"][:len(steps)] for q in e["query_rows"]}),
                "real_held_presentations": len(holds), "canonical_presentations": len(canonical),
                "canonical_products": len(c_products), "canonical_refs_used": len(ref_used),
                "canonical_last_batch_size": len(steps[-1]["canonical"]), "counts": dict(counts),
                "canonical_weighted_producer_share": {p: weighted_producers[p]/len(canonical) for p in sorted(weighted_producers)},
                "canonical_unweighted_producer_share": dict(Counter(t["producer"] for t in canonical)),
                "canonical_producer_product_counts": {p:len(v) for p,v in sorted(producer_products.items())},
                "canonical_product_exposure": _stats(list(c_products.values())),
                "canonical_weighted_negative_product_exposure": _stats(list(weighted_negative_products.values())),
                "canonical_weighted_negative_product_mass_total":sum(weighted_negative_products.values()),
                "real_query_exposure": _stats(list(Counter(x["query_id"] for s in steps for x in s["real"]).values())),
                "real_query_views_without_effective_negative_loss":len(no_effect),
                "real_query_no_effect_ids":no_effect,
                "real_query_forward_inbatch_cells":sum(x.get("forward_inbatch_negative_cells",0) for x in query_loss_rows),
                "real_query_reverse_inbatch_cells":sum(x.get("reverse_inbatch_negative_cells",0) for x in query_loss_rows),
                "real_query_forward_mined_cells":sum(x.get("forward_mined_negative_cells",0) for x in query_loss_rows)}
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    write_once(OUT / "steps.jsonl", jsonl_bytes(steps))
    write_once(OUT / "real-holds.jsonl", jsonl_bytes(holds))
    write_once(OUT / "real-query-loss-coverage.jsonl", jsonl_bytes(query_loss_rows))
    write_once(OUT / "coverage.json", canonical_bytes(coverage))
    pins = {str(p.relative_to(ROOT)):sha256(p) for p in (STAGE3/"manifest.json",STAGE4/"manifest.json",REAL,OLD,
            OLD_RECIPE,REFS,REG01,REG03,SOURCE_LEDGER,MASK,NEGATIVE_MANIFEST,OLD_LEDGER)}
    runtime_pins = {str(p.relative_to(ROOT)):sha256(p) for p in (REAL,REFS,NEGATIVE_MANIFEST)}
    code_pins = {rel:sha256(ROOT/rel) for rel in (
        "rshb_vine/__init__.py",
        "rshb_vine/catalog_training_v2/__init__.py",
        "rshb_vine/catalog_training_v2/prepare.py",
        "rshb_vine/catalog_training_v2/data.py",
        "rshb_vine/catalog_training_v2/loss.py",
        "rshb_vine/catalog_training_v2/trainer.py",
        "rshb_vine/metric_adaptation.py",
        "scripts/catalog_training_v2.py",
        "rshb_vine/domain_augmentation.py", "rshb_vine/preprocessing/__init__.py", "rshb_vine/io.py")}
    manifest = {"schema_version":"catalog-training-v2-stage5-consumer-v1", "status":"prepared_not_fit_admitted",
                "fit_admitted":False, "canonical_per_batch_proposal":4, "real_per_batch_max":4,
                "steps":len(steps), "temperature":read(OLD_RECIPE)["temperature"],
                "loss_weights":{"real_global":1.0,"real_local":0.25,"canonical_triplet":0.0625},
                "parent_encoder":read(OLD_RECIPE)["parent_encoder"], "parent_files_sha256":read(OLD_RECIPE)["parent_files_sha256"],
                "optimizer_control":{"learning_rate":read(OLD_RECIPE)["learning_rate"],"weight_decay":read(OLD_RECIPE)["weight_decay"],
                                     "lora_rank":read(OLD_RECIPE)["lora_rank"],"lora_alpha":read(OLD_RECIPE)["lora_alpha"],
                                     "lora_targets":read(OLD_RECIPE)["lora_targets"]},
                "input_pins":pins, "runtime_input_pins":runtime_pins, "code_pins":code_pins,
                "real_augmentation_seed":read(OLD_RECIPE)["seed"],
                "precision":read(OLD_RECIPE)["precision"], "old_recipe_schema":read(OLD_RECIPE)["schema_version"],
                "output_pins":{n:sha256(OUT/n) for n in ("steps.jsonl","real-holds.jsonl","real-query-loss-coverage.jsonl","coverage.json")},
                "stage34_admission_sha256":sha256(preparation.ADMISSION),
                "rules":{"real_unknown_pairs":"masked unless every acceptable current ID combination has current stage3 admitted/real-role semantic negative evidence or, where stage3 has no explicit decision, explicit old admitted ordinary/hard/query-specific mined ledger evidence; explicit stage3 masks/holds and same-source/same-ID always masked",
                         "canonical":"independent (anchor, positive, nominated negative) triplets; positive is same-source or root-admitted same-ProductID cross-source (all three source ends admitted)",
                         "cross_branch":"no cross-branch comparisons", "product_weight":"equal total canonical anchor mass per ProductID per cycle",
                         "batch":"4 real max, 4 canonical proposed, final canonical batch partial",
                         "prior_real_negatives":"old admitted ordinary/hard and query-specific mined ledger only where stage3 has no explicit hold",
                         "real_role_semantics":"stage3 evidenced negative relation may serve admitted real query when canonical source unavailable; actual negative source must be admitted"}}
    write_once(OUT / "manifest.json", canonical_bytes(manifest))
    receipt = {"status":"cpu_stage5_prepared", "fit_admitted":False, "manifest_sha256":sha256(OUT/"manifest.json"),
               "steps_sha256":sha256(OUT/"steps.jsonl"), "coverage":coverage,
               "reference_indices_used":len(ref_used), "reference_sources_inventory":len(sources),
               "negative_policy":"real negatives: current stage3 admitted or real-role semantic negative edge, else (no stage3 decision) explicit old admitted ordinary/hard/query-specific mined ledger; actual negative source must be stage2 admitted; explicit stage3 masks/holds prevail",
               "real_query_unique_source_sha256_retained":len({x["query_source_sha256"] for s in steps for x in s["real"]}),
               "real_reference_unique_source_sha256_retained":len({x["reference_source_sha256"] for s in steps for x in s["real"]}),
               "real_query_unique_source_sha256_old_cycle":len({row_by_id[q]["source_sha256"] for e in old["steps"][:len(steps)] for q in e["query_rows"]})}
    write_once(RUN / "receipt.json", canonical_bytes(receipt))
    return receipt


def validate_seal():
    m = read(OUT / "manifest.json")
    if m.get("schema_version") != "catalog-training-v2-stage5-consumer-v1" or m.get("fit_admitted") is not False:
        raise ValueError("Unexpected stage5 manifest")
    for rel,digest in m["input_pins"].items():
        if sha256(ROOT/rel) != digest:
            raise ValueError("Stage5 input drift: "+rel)
    for rel,digest in m["code_pins"].items():
        if sha256(ROOT/rel) != digest:
            raise ValueError("Stage5 code drift: "+rel)
    for name,digest in m["output_pins"].items():
        if sha256(OUT/name) != digest:
            raise ValueError("Stage5 output drift: "+name)
    return m


def validate_runtime_seal():
    """Portable validation of every file needed for batch loading/loss."""
    m = read(OUT/"manifest.json")
    if m.get("schema_version") != "catalog-training-v2-stage5-consumer-v1" or m.get("fit_admitted") is not False or \
            m.get("canonical_per_batch_proposal") != 4:
        raise ValueError("Unexpected stage5 runtime manifest")
    for rel,digest in {**m["runtime_input_pins"],**m["code_pins"]}.items():
        if sha256(ROOT/rel) != digest:
            raise ValueError("Stage5 runtime pin drift: "+rel)
    for name,digest in m["output_pins"].items():
        if sha256(OUT/name) != digest:
            raise ValueError("Stage5 output drift: "+name)
    parent = ROOT/m["parent_encoder"]
    for name,digest in m["parent_files_sha256"].items():
        if sha256(parent/name) != digest:
            raise ValueError("Parent checkpoint drift: "+name)
    return m
