"""Guarded stage5 LoRA trainer: non-updating profile, one bounded fit, explicit resume, local export.

The trainer never creates, stops or deletes a provider resource. Every paid mode
fails before the model or optimizer exists unless the frozen recipe, the pre-start
root admission/spend, the post-start lease and (for fit) the same-lease profile
all bind the sealed stage5 inputs.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from rshb_vine.catalog_training_v2 import prepare as P
from rshb_vine.io import sha256

RECIPE_SCHEMA = "catalog-training-v2-stage5-recipe-v1"
RECIPE_STATUS = "frozen_catalog_training_v2_stage5_recipe"
ADMISSION_STATUS = "root_admitted_catalog_training_v2_stage5_fit"
SPEND_STATUS = "root_authorized_spend_catalog_training_v2_stage5"
LEASE_STATUS = "root_bound_resource_lease_catalog_training_v2_stage5"
RESUME_STATUS = "root_authorized_resume_catalog_training_v2_stage5"
PROFILE_KIND = "catalog-training-v2-stage5-cuda-profile"
OLD_RECIPE_SCHEMA = "visual-learning-v1-corrected-b3-recipe"
PROFILE_STEPS, PROFILE_WARMUP = 64, 16
MAX_CYCLES = 2
RESERVE_SECONDS = 20*60
EVIDENCE = ("source_role_and_protected_audit", "validation_exposure_audit",
            "extracted_linux_transport_and_cpu_preflight_receipt")
SPEND_CAPS = {"max_usd": 2.0, "max_lease_minutes": 90, "max_all_in_quote_usd_per_hour": 0.9,
              "min_non_gpu_reserve_usd": 0.5}


def read(path):
    return json.loads(Path(path).read_text())


def utc(text):
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Deadline/lease time must be absolute UTC")
    return value.timestamp()


def code_pins_digest(manifest):
    return hashlib.sha256(P.canonical_bytes(manifest["code_pins"])).hexdigest()


def expected_recipe(manifest):
    """Every trainer factor derived from the sealed stage5 manifest and old B3 recipe."""
    if "rshb_vine/catalog_training_v2/trainer.py" not in manifest["code_pins"]:
        raise ValueError("Stage5 seal does not pin the trainer")
    steps = manifest["steps"]
    canonical = read(P.OUT/"coverage.json")["canonical_presentations"]
    per_step = manifest["canonical_per_batch_proposal"]
    if math.ceil(canonical/per_step) != steps:
        raise ValueError("Canonical cycle/step count mismatch")
    half = math.ceil(canonical/2)
    targets = [half, canonical, canonical+half, 2*canonical]
    step_targets = [math.ceil(half/per_step), steps, steps+math.ceil(half/per_step), 2*steps]
    if manifest.get("old_recipe_schema") != OLD_RECIPE_SCHEMA:
        raise ValueError("Old B3 recipe summary drift")
    return {
        "schema_version": RECIPE_SCHEMA, "status": RECIPE_STATUS,
        "stage5_manifest_sha256": sha256(P.OUT/"manifest.json"),
        "stage5_steps_sha256": manifest["output_pins"]["steps.jsonl"],
        "stage5_consumer_receipt_sha256": sha256(P.RUN/"receipt.json"),
        "cache_receipt_sha256": sha256(P.OUT/"cache/receipt.json"),
        "stage4_manifest_sha256": manifest["input_pins"]["data/catalog-training-data-v2/stage4/manifest.json"],
        "source_role_decisions_sha256": manifest["input_pins"]["data/catalog-training-data-v2/stage2/source-ledger.jsonl"],
        "code_pins_sha256": code_pins_digest(manifest),
        "parent_encoder": manifest["parent_encoder"],
        "parent_encoder_files_sha256": manifest["parent_files_sha256"],
        "old_recipe_sha256": manifest["input_pins"]["runs/visual-learning-v1/trial/source-hold-v2/recipe-fit.json"],
        "seed": manifest["real_augmentation_seed"],
        "learning_rate": manifest["optimizer_control"]["learning_rate"],
        "weight_decay": manifest["optimizer_control"]["weight_decay"],
        "optimizer": "AdamW over LoRA parameters only", "grad_clip_norm": 1.0,
        "lora_rank": manifest["optimizer_control"]["lora_rank"],
        "lora_alpha": manifest["optimizer_control"]["lora_alpha"],
        "lora_targets": manifest["optimizer_control"]["lora_targets"],
        "precision": manifest["precision"], "temperature": manifest["temperature"],
        "loss_coefficients": manifest["loss_weights"],
        "product_anchor_weight_rule": "w=N/(P*n_p) per canonical anchor ProductID; fixed four-slot denominator",
        "canonical_triplets_per_step": manifest["canonical_per_batch_proposal"],
        "real_anchors_per_step": manifest["real_per_batch_max"],
        "masked_partial_slots": "final canonical batch of each cycle keeps its explicit empty slots",
        "steps_per_cycle": steps, "max_canonical_cycles": MAX_CYCLES,
        "cycle_rule": "later cycle replays sealed steps in order; real query augmentation index=global_step*4+old_position",
        "checkpoint_presentation_targets": targets, "checkpoint_steps": step_targets,
        "selection_eligible_checkpoint_steps": step_targets[1:], "diagnostic_only_checkpoint_steps": step_targets[:1],
        "max_optimizer_steps": step_targets[-1],
        "profile_steps": PROFILE_STEPS, "profile_warmup_steps": PROFILE_WARMUP,
        "reserve_seconds_download_stop": RESERVE_SECONDS,
    }


def verify_recipe(path, manifest):
    recipe = read(path)
    expected = expected_recipe(manifest)
    diff = sorted(k for k in set(recipe) | set(expected) if recipe.get(k) != expected.get(k))
    if diff:
        raise ValueError("Recipe differs from sealed stage5 factors: " + ",".join(diff))
    return recipe


def _need(doc, name, **equal):
    for key, value in equal.items():
        if doc.get(key) != value:
            raise ValueError(f"{name} does not bind {key}")


def verify_gates(mode, args, manifest, recipe_sha):
    """All paid-mode receipts, checked before torch/model/optimizer creation."""
    for flag in ("admission", "spend", "lease"):
        if getattr(args, flag) is None or not Path(getattr(args, flag)).is_file():
            raise ValueError(f"{mode} requires an existing --{flag} receipt")
    admission, spend, lease = read(args.admission), read(args.spend), read(args.lease)
    _need(admission, "Admission", status=ADMISSION_STATUS, exact_recipe_sha256=recipe_sha,
          stage5_manifest_sha256=sha256(P.OUT/"manifest.json"), profile_before_optimizer=True,
          one_fit_only=True, final_stage5_consumer_receipt_sha256=sha256(P.RUN/"receipt.json"),
          fit_output=str(Path(args.output).resolve()) if mode == "fit" else admission.get("fit_output"),
          profile_output=str(Path(args.output).resolve()) if mode == "profile" else admission.get("profile_output"))
    if admission["fit_output"] == admission["profile_output"]:
        raise ValueError("Admission must fix distinct profile and fit outputs")
    evidence = admission.get("evidence") or {}
    for key in EVIDENCE:
        item = evidence.get(key) or {}
        path = P.ROOT/str(item.get("path", ""))
        if not item.get("path") or not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError("Admission evidence absent or SHA differs: " + key)
    _need(spend, "Spend", status=SPEND_STATUS, exact_root_fit_admission_sha256=sha256(args.admission),
          exact_recipe_sha256=recipe_sha, one_fit_maximum=True)
    if not spend.get("resource_intent_nonce"):
        raise ValueError("Spend lacks resource intent nonce")
    if not (0 < spend["fresh_all_in_quote_usd_per_hour"] <= SPEND_CAPS["max_all_in_quote_usd_per_hour"]
            and 0 < spend["max_usd"] <= SPEND_CAPS["max_usd"]
            and 0 < spend["max_lease_minutes"] <= SPEND_CAPS["max_lease_minutes"]
            and spend["non_gpu_reserve_usd"] >= SPEND_CAPS["min_non_gpu_reserve_usd"]):
        raise ValueError("Spend exceeds proposed caps")
    deadline = utc(spend["absolute_deadline_utc"])
    _need(lease, "Lease", status=LEASE_STATUS, spend_sha256=sha256(args.spend),
          resource_intent_nonce=spend["resource_intent_nonce"], absolute_deadline_utc=spend["absolute_deadline_utc"])
    resource = lease.get("resource_id")
    if not resource or os.environ.get("RUNPOD_POD_ID") != resource:
        raise ValueError("Lease resource ID differs from this host")
    started = utc(lease["resource_started_utc"])
    # Money is a clock at the quoted all-in rate; the non-GPU reserve is never spent on GPU time.
    money_seconds = (spend["max_usd"] - spend["non_gpu_reserve_usd"])/spend["fresh_all_in_quote_usd_per_hour"]*3600
    deadline = min(deadline, started + spend["max_lease_minutes"]*60, started + money_seconds)
    if time.time() + RESERVE_SECONDS >= deadline:
        raise ValueError("No time left before absolute deadline reserve")
    return {"admission": admission, "spend": spend, "lease": lease, "deadline": deadline, "started": started,
            "admission_sha256": sha256(args.admission), "spend_sha256": sha256(args.spend),
            "lease_sha256": sha256(args.lease)}


def verify_profile(args, gates, recipe_sha, recipe):
    if args.profile is None or not Path(args.profile).is_file() or \
            Path(args.profile).resolve() != Path(gates["admission"]["profile_output"])/"profile.json":
        raise ValueError("fit requires the same-lease CUDA profile receipt from the admitted profile output")
    profile = read(args.profile)
    _need(profile, "Profile", kind=PROFILE_KIND, status="profile_passed", device="cuda",
          recipe_sha256=recipe_sha, lease_sha256=gates["lease_sha256"],
          resource_id=gates["lease"]["resource_id"], optimizer_steps=0, checkpoint_written=False,
          steps=PROFILE_STEPS)
    seconds = profile["sustained_seconds_per_step"]
    if not (isinstance(seconds, (int, float)) and seconds > 0):
        raise ValueError("Profile lacks sustained step time")
    projected = recipe["max_optimizer_steps"]*seconds
    spend = gates["spend"]
    finish = time.time() + projected + RESERVE_SECONDS
    cost = (finish - gates["started"])/3600*spend["fresh_all_in_quote_usd_per_hour"]
    if finish > gates["deadline"]:
        raise ValueError(f"Projected fit {projected:.0f}s + reserve exceeds deadline/cost cap; stop GPU without fit")
    return {"profile_sha256": sha256(args.profile), "projected_fit_seconds": projected,
            "projected_all_in_usd": cost}


def verify_resume(args, gates, recipe_sha, checkpoint):
    if args.resume_receipt is None or not Path(args.resume_receipt).is_file():
        raise ValueError("Resume requires an explicit root resume receipt")
    receipt = read(args.resume_receipt)
    _need(receipt, "Resume", status=RESUME_STATUS, exact_recipe_sha256=recipe_sha,
          admission_sha256=gates["admission_sha256"], spend_sha256=gates["spend_sha256"],
          lease_sha256=gates["lease_sha256"], checkpoint_sha256=sha256(checkpoint),
          fit_claim_sha256=sha256(Path(args.output)/"fit-claim.json"),
          absolute_deadline_utc=gates["spend"]["absolute_deadline_utc"])
    return sha256(args.resume_receipt)


def _checkpoint(path, state):
    import torch
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return sha256(path)


def run(args):
    """profile|fit. Gates are complete before importing torch or loading the parent."""
    manifest = P.validate_runtime_seal()
    recipe = verify_recipe(args.recipe, manifest)
    recipe_sha = sha256(args.recipe)
    gates = verify_gates(args.mode, args, manifest, recipe_sha)
    output = Path(args.output)
    profile_gate = resume_sha = None
    checkpoint_path = output/"state-latest.pt"
    if args.mode == "profile":
        if args.resume or (output.exists() and any(output.iterdir())):
            raise ValueError("Profile output must be new and empty; profile never resumes")
    else:
        profile_gate = verify_profile(args, gates, recipe_sha, recipe)
        if (output/"result.json").exists():
            raise ValueError("Completed fit is immutable")
        if args.resume:
            if not checkpoint_path.is_file() or not (output/"fit-claim.json").is_file():
                raise ValueError("Resume checkpoint or single-fit claim absent")
            claim = read(output/"fit-claim.json")
            if claim.get("lease_sha256") != gates["lease_sha256"] or claim.get("admission_sha256") != gates["admission_sha256"]:
                raise ValueError("Single-fit claim belongs to another lease/admission")
            resume_sha = verify_resume(args, gates, recipe_sha, checkpoint_path)
        elif output.exists() and any(output.iterdir()):
            raise ValueError("Fit output exists; only an explicit root resume receipt may continue it")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from rshb_vine.catalog_training_v2.data import PreparedSteps
    from rshb_vine.catalog_training_v2.loss import combined_loss
    from rshb_vine.metric_adaptation import install_vision_lora

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 required by the frozen recipe")
    device = "cuda"
    dataset = PreparedSteps(portable=True)
    if len(dataset) != recipe["steps_per_cycle"]:
        raise ValueError("Prepared step count differs from recipe")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "fit" and not args.resume:
        claim = json.dumps({"admission_sha256": gates["admission_sha256"], "lease_sha256": gates["lease_sha256"],
                            "fit_output": str(output.resolve()), "claimed_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        fd = os.open(output/"fit-claim.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w") as handle:
            handle.write(claim)
    trial = {"recipe_sha256": recipe_sha, "stage5_manifest_sha256": recipe["stage5_manifest_sha256"],
             "admission_sha256": gates["admission_sha256"], "spend_sha256": gates["spend_sha256"],
             "lease_sha256": gates["lease_sha256"], "resource_id": gates["lease"]["resource_id"],
             "trainer_sha256": sha256(Path(__file__)), "hardware": torch.cuda.get_device_name(0),
             "torch": torch.__version__, "mode": args.mode}
    trial_path = output/("profile-trial.json" if args.mode == "profile" else "trial.json")
    if trial_path.exists() and read(trial_path) != trial:
        raise ValueError("Trial identity changed")
    trial_path.write_text(json.dumps(trial, indent=2), encoding="utf-8")

    torch.manual_seed(recipe["seed"])
    model = AutoModel.from_pretrained(P.ROOT/recipe["parent_encoder"], local_files_only=True).to(device)
    lora = install_vision_lora(model)
    if (lora["rank"], lora["alpha"]) != (recipe["lora_rank"], recipe["lora_alpha"]):
        raise ValueError("LoRA helper differs from recipe")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    start = 0
    if args.mode == "fit":
        optimizer = torch.optim.AdamW(trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        if args.resume:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state["trial_recipe_sha256"] != recipe_sha or state["resource_id"] != trial["resource_id"] or \
                    not 0 < state["global_step"] < recipe["max_optimizer_steps"]:
                raise ValueError("Checkpoint belongs to another recipe/resource or invalid step")
            named = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in state["lora"].items():
                    named[name].copy_(value)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            start = state["global_step"]
    last = PROFILE_STEPS if args.mode == "profile" else recipe["max_optimizer_steps"]

    class Cycles(torch.utils.data.Dataset):
        def __len__(self):
            return last - start

        def __getitem__(self, index):
            return dataset[start+index]

    # Own generator: creating the iterator must not consume the restored global RNG stream.
    loader = DataLoader(Cycles(), batch_size=None, num_workers=args.workers,
                        prefetch_factor=2 if args.workers else None, pin_memory=True,
                        persistent_workers=bool(args.workers),
                        generator=torch.Generator().manual_seed(recipe["seed"]+start))
    log = (output/("profile-steps.jsonl" if args.mode == "profile" else "steps.jsonl")).open("a", encoding="utf-8")
    model.train()
    checkpoints = set(recipe["checkpoint_steps"])
    wait = compute = 0.0
    walls, exposures = [], Counter()
    started = time.monotonic()
    fetched = previous = time.monotonic()
    step = start
    in_update = False
    status = "running"

    def save_state(kind):
        state = {"trial_recipe_sha256": recipe_sha, "resource_id": trial["resource_id"], "global_step": step,
                 "lora": {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad},
                 "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all(), "kind": kind}
        digest = _checkpoint(checkpoint_path, state)
        if kind == "target":
            target = output/f"checkpoint-{step:06d}.pt"
            if target.exists():
                raise ValueError("Target checkpoint already exists")
            os.link(checkpoint_path, target)
        return digest

    try:
        for batch in loader:
            wait += time.monotonic() - fetched
            t0 = time.monotonic()
            if time.time() + RESERVE_SECONDS >= gates["deadline"]:
                status = "stopped_deadline_reserve"
                break
            pixels = batch["pixels"].to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.get_image_features(pixel_values=pixels)
            parts = combined_loss(features.float(), batch, temperature=recipe["temperature"],
                                  real_global_weight=recipe["loss_coefficients"]["real_global"],
                                  real_local_weight=recipe["loss_coefficients"]["real_local"],
                                  canonical_weight=recipe["loss_coefficients"]["canonical_triplet"])
            parts["total"].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, recipe["grad_clip_norm"],
                                                             error_if_nonfinite=True))
            if optimizer is not None:
                in_update = True
                optimizer.step()
                in_update = False
            torch.cuda.synchronize()
            step = batch["global_step"] + 1
            done = time.monotonic()
            seconds = done - t0
            compute += seconds
            walls.append(done - previous)
            previous = done
            exposures.update(x["anchor"]["product03"] for x in batch["step"]["canonical"])
            row = {"global_step": step, "cycle": batch["cycle"], "loss": {k: float(v.detach()) for k, v in parts.items()},
                   "grad_norm_before_clip": grad_norm, "real": batch["real_count"], "canonical": batch["canonical_count"],
                   "mined": batch["mined_count"], "compute_seconds": seconds, "wall_seconds": walls[-1],
                   "images_per_second": batch["pixels"].shape[0]/seconds}
            log.write(json.dumps(row, sort_keys=True)+"\n")
            if optimizer is not None and step in checkpoints:
                digest = save_state("target")
                log.flush()
                (output/f"progress-{step:06d}.json").write_text(json.dumps(
                    {"global_step": step, "checkpoint_sha256": digest, "elapsed_seconds": time.monotonic()-started,
                     "data_wait_seconds": wait, "compute_seconds": compute,
                     "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                     "diagnostic_only": step in recipe["diagnostic_only_checkpoint_steps"]}, indent=2), encoding="utf-8")
            fetched = time.monotonic()
        else:
            status = "completed"
    except BaseException as error:
        status = "failed"
        (output/f"failure-{int(time.time())}.json").write_text(json.dumps(
            {"global_step": step, "error": repr(error), "in_optimizer_update": in_update}, indent=2), encoding="utf-8")
        if optimizer is not None and not in_update and step > start:
            save_state("failure_last_completed_step")
        raise
    finally:
        log.close()
    elapsed = time.monotonic() - started
    if args.mode == "profile":
        sustained = walls[PROFILE_WARMUP:]
        report = {"kind": PROFILE_KIND, "status": "profile_passed" if status == "completed" else status,
                  "device": device, "steps": step, "warmup_steps_excluded": PROFILE_WARMUP,
                  "workers": args.workers, "recipe_sha256": recipe_sha, "lease_sha256": gates["lease_sha256"],
                  "resource_id": trial["resource_id"], "optimizer_steps": 0, "checkpoint_written": False,
                  "sustained_seconds_per_step": sum(sustained)/len(sustained) if sustained else None,
                  "sustained_rule": "mean wall time incl. data wait and CUDA sync after warmup",
                  "data_wait_seconds": wait, "compute_seconds": compute, "elapsed_seconds": elapsed,
                  "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                  "note": "forward/backward/clip only; LoRA parameters never updated"}
        (output/"profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        return report
    if status != "completed":
        digest = save_state("deadline_reserve")
        stop = {"status": status, "global_step": step, "state_sha256": digest, "resume_requires": RESUME_STATUS}
        (output/"stopped.json").write_text(json.dumps(stop, indent=2), encoding="utf-8")
        print(json.dumps(stop), flush=True)
        return stop
    if step != recipe["max_optimizer_steps"]:
        raise ValueError("Incomplete fixed budget")
    result = {"status": "fit_complete_unreleased", "global_step": step, "trial": trial, **profile_gate,
              "resume_receipt_sha256": resume_sha,
              "checkpoints": {str(s): sha256(output/f"checkpoint-{s:06d}.pt") for s in recipe["checkpoint_steps"]
                              if (output/f"checkpoint-{s:06d}.pt").exists()},
              "segment_canonical_anchor_exposure": dict(sorted(exposures.items())),
              "export": "merge LoRA locally with catalog_training_v2 export; gallery reindex required",
              "release_ready": False}
    (output/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "global_step", "checkpoints")}), flush=True)
    return result


def export(checkpoint, output, recipe_path):
    """Local CPU merge of a downloaded target checkpoint into a plain encoder directory."""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from rshb_vine.metric_adaptation import install_vision_lora, merge_vision_lora
    manifest = P.validate_runtime_seal()
    recipe = verify_recipe(recipe_path, manifest)
    output = Path(output)
    if output.exists():
        raise ValueError("Export output exists")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("trial_recipe_sha256") != sha256(recipe_path) or state.get("kind") != "target" or \
            state.get("global_step") not in recipe["checkpoint_steps"]:
        raise ValueError("Checkpoint is not a target checkpoint of this frozen recipe")
    parent = P.ROOT/manifest["parent_encoder"]
    model = AutoModel.from_pretrained(parent, local_files_only=True)
    install_vision_lora(model)
    named = dict(model.named_parameters())
    if set(state["lora"]) != {k for k, p in named.items() if p.requires_grad}:
        raise ValueError("Checkpoint LoRA parameter set differs")
    with torch.no_grad():
        for name, value in state["lora"].items():
            named[name].copy_(value)
    merge_vision_lora(model)
    model.save_pretrained(output, safe_serialization=True)
    AutoImageProcessor.from_pretrained(parent, local_files_only=True, use_fast=False).save_pretrained(output)
    return {"status": "exported_unreleased", "global_step": state["global_step"],
            "diagnostic_only": state["global_step"] in recipe["diagnostic_only_checkpoint_steps"],
            "checkpoint_sha256": sha256(checkpoint), "recipe_sha256": state["trial_recipe_sha256"],
            "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}, "gallery_reindex_required": True}
