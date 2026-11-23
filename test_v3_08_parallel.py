"""Isolation and stage-gate tests for the parallel 4v2 experiment."""

import hashlib
import hmac
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


def _parallel():
    return importlib.import_module(f"{__package__}.v3_08_parallel")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _registry():
    return {
        "protocol": {"name": "deploy5x5", "d_cap": 0.8,
                     "pursuer_speed": 0.35, "evader_speed": 0.35},
        "training_pool": [72000, 72999],
        "validation_pool": [73000, 73199],
        "generalization_pool": [74000, 74199],
        "sealed_seed": 70086,
        "stage_seeds": {"ring_expert": 41, "counterfactual": 43,
                        "feasibility": 47, "actor": 53, "allocation": 59},
        "selection": {"ring_expert": {"primary": "full_capture_rate", "maximize": True,
                                      "tie_break": "mean_capture_time"}},
        "allocations": {
            "ring_expert": {"0": [72000, 72001], "1": [72002, 72003]},
            "counterfactual": {"0": [72000, 72001], "1": [72002, 72003]},
        },
    }


@pytest.fixture
def context(tmp_path, monkeypatch):
    parallel = _parallel()
    root = tmp_path / "output"
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.py").write_text("ONE = 1\n")
    (source / "two.py").write_text("TWO = 2\n")
    baseline = tmp_path / "baseline.pth"
    baseline.write_bytes(b"authoritative-baseline")
    registry = tmp_path / "seed_registry.json"
    registry.write_text(json.dumps(_registry()))
    monkeypatch.setattr(parallel, "OUTPUT_ROOT", root)
    monkeypatch.setattr(parallel, "SOURCE_ROOT", source, raising=False)
    monkeypatch.setattr(parallel, "BASELINE_PATH", baseline, raising=False)
    monkeypatch.setattr(parallel, "SEED_REGISTRY_PATH", registry)
    monkeypatch.setenv("V3_08_COORDINATOR_TOKEN", "coordinator-private-test-token")
    monkeypatch.setenv("V3_08_EVALUATOR_KEY", "evaluator-private-test-key")
    return {"parallel": parallel, "root": root, "source": source, "baseline": baseline,
            "registry": registry, "frozen": None}


def _freeze(context):
    if context["frozen"] is None:
        context["frozen"] = context["parallel"].freeze_inputs()
    return context["frozen"]


def _arm(name="expert"):
    return {"name": name, "argv": [sys.executable, "-c", "print('worker complete')"]}


def _worker(context, worker_id="0", *, name="expert", artifact_bytes=b"checkpoint-v1"):
    _freeze(context)
    parallel = context["parallel"]
    run = parallel.create_run("ring_expert", _arm(name), worker_id)
    artifact = run.directory / "checkpoint.bin"
    artifact.write_bytes(artifact_bytes)
    sample_seed = run.seeds["train"][0]
    samples = [{"scenario_seed": sample_seed, "origin_seed": sample_seed,
                "scenario_id": f"train-{sample_seed}"}]
    sample_path = run.directory / "sample_manifest.json"
    sample_path.write_text(json.dumps({"samples": samples}))
    payload = {
        "stage": "ring_expert", "arm": name, "worker_id": worker_id,
        "run_id": run.run_id, "status": "complete", "seeds": run.seeds,
        "samples": samples,
        "sample_manifest": {"path": "sample_manifest.json", "sha256": _hash(sample_path)},
        "source_sha256": context["frozen"]["source_files"],
        "source_tree_sha256": context["frozen"]["source_tree_sha256"],
        "baseline_sha256": context["frozen"]["baseline"]["sha256"],
        "registry_sha256": context["frozen"]["seed_registry"]["sha256"],
        "artifact": {"path": "checkpoint.bin", "sha256": _hash(artifact)},
    }
    path = run.directory / "artifact_manifest.json"
    path.write_text(json.dumps(payload))
    parallel.write_status(run.directory / "status.json", {"state": "complete", "run_id": run.run_id})
    return run, path, payload


def _evaluation(context, workers):
    frozen = context["frozen"]
    payload = {
        "stage": "ring_expert", "protocol": _registry()["protocol"],
        "evaluation_seeds": list(range(74000, 74200)),
        "source_tree_sha256": frozen["source_tree_sha256"],
        "baseline": {"checkpoint_sha256": frozen["baseline"]["sha256"],
                     "seeds": list(range(74000, 74200)),
                     "metrics": {"full_capture_rate": 0.2, "mean_capture_time": 40.0}},
        "candidates": [
            {"run_id": run.run_id, "manifest_path": str(path),
             "checkpoint_sha256": data["artifact"]["sha256"],
             "source_tree_sha256": frozen["source_tree_sha256"],
             "seeds": list(range(74000, 74200)),
             "metrics": {"full_capture_rate": score, "mean_capture_time": 30.0}}
            for (run, path, data), score in zip(workers, (0.8, 0.6))
        ],
        "winner_run_id": workers[0][0].run_id,
        "fallback_run_id": workers[1][0].run_id,
    }
    _sign(payload)
    path = context["root"] / "evaluation" / "ring_expert" / "gate-1" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    return path, payload


def _sign(payload):
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = hmac.new(b"evaluator-private-test-key", raw, hashlib.sha256).hexdigest()


def test_freeze_records_complete_source_baseline_registry_and_protocol(context):
    frozen = _freeze(context)
    assert set(frozen["source_files"]) == {"one.py", "two.py"}
    assert frozen["source_files"]["one.py"] == _hash(context["source"] / "one.py")
    assert frozen["baseline"]["sha256"] == _hash(context["baseline"])
    assert frozen["seed_registry"]["sha256"] == _hash(context["registry"])
    assert frozen["protocol"] == _registry()["protocol"]
    assert json.loads((context["root"] / "coordination" / "frozen_source.json").read_text()) == frozen


def test_freeze_can_refresh_changed_source_before_any_run(context):
    original = _freeze(context)
    (context["source"] / "two.py").write_text("TWO = 3\n")
    updated = context["parallel"].freeze_inputs(refresh=True)
    assert updated["source_tree_sha256"] != original["source_tree_sha256"]


@pytest.mark.parametrize("target", ["source", "baseline", "registry"])
def test_create_run_blocks_drift_of_any_frozen_input(context, target):
    _freeze(context)
    path = context[target]
    if path.is_dir():
        (path / "two.py").write_text("TWO = 3\n")
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="frozen|hash|source"):
        context["parallel"].create_run("ring_expert", _arm(), "0")


def test_run_reserves_unique_directory_and_launch_contract(context):
    _freeze(context)
    parallel = context["parallel"]
    run = parallel.create_run("ring_expert", _arm(), "0")
    other = parallel.create_run("ring_expert", _arm(), "1")
    config = json.loads((run.directory / "config.json").read_text())
    assert run.directory == context["root"] / "datasets" / "ring_expert" / "shard_0"
    assert run.command[0] == sys.executable
    assert str(run.directory / "config.json") in run.command
    assert (run.directory / "worker.pid").exists()
    assert (run.directory / "tmp").is_dir()
    assert config["resources"]["cpu_affinity"] != json.loads((other.directory / "config.json").read_text())["resources"]["cpu_affinity"]
    assert config["resources"]["omp_threads"] >= 1
    assert config["resources"]["gpu_shared"] is True
    assert len(set(config["rng_seeds"].values())) == len(config["rng_seeds"])
    assert config["source_tree_sha256"] == context["frozen"]["source_tree_sha256"]
    with pytest.raises(FileExistsError):
        parallel.create_run("ring_expert", _arm(), "0")


def test_launch_command_runs_worker_with_isolated_pid_log_and_temp(context):
    _freeze(context)
    run = context["parallel"].create_run("ring_expert", _arm(), "0")
    completed = subprocess.run(run.command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert int((run.directory / "worker.pid").read_text()) > 0
    assert "worker complete" in (run.directory / "train.log").read_text()
    assert json.loads((run.directory / "status.json").read_text())["state"] == "complete"


def test_manifest_requires_reserved_run_identity_and_config(context):
    parallel = context["parallel"]
    run, path, payload = _worker(context)
    assert parallel.validate_worker_manifest(path, context["registry"])["run_id"] == run.run_id
    forged = context["root"] / "forged" / "artifact_manifest.json"
    forged.parent.mkdir()
    forged.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reserved|directory"):
        parallel.validate_worker_manifest(forged, context["registry"])
    config = json.loads((run.directory / "config.json").read_text())
    config["arm"] = "imposter"
    (run.directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config"):
        parallel.validate_worker_manifest(path, context["registry"])


def test_manifest_rejects_overlapping_allocations(context):
    _, path, _ = _worker(context)
    registry = json.loads(context["registry"].read_text())
    registry["allocations"]["ring_expert"]["1"] = [72001, 72003]
    context["registry"].write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="hash|overlap"):
        context["parallel"].validate_worker_manifest(path, context["registry"])


@pytest.mark.parametrize("leak", [{"origin_seed": 70086}, {"scenario_id": "70086-jitter-1"}])
def test_manifest_rejects_sealed_or_derived_sample(context, leak):
    _, path, payload = _worker(context)
    payload["samples"][0].update(leak)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sealed|sample"):
        context["parallel"].validate_worker_manifest(path, context["registry"])


def test_manifest_rejects_missing_origin_and_sample_content_hash_mismatch(context):
    _, path, payload = _worker(context)
    del payload["samples"][0]["origin_seed"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="origin|sample"):
        context["parallel"].validate_worker_manifest(path, context["registry"])
    payload["samples"][0]["origin_seed"] = 72000
    payload["sample_manifest"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sample.*hash|hash.*sample"):
        context["parallel"].validate_worker_manifest(path, context["registry"])


@pytest.mark.parametrize("field", ["artifact", "source_sha256"])
def test_manifest_rejects_artifact_or_frozen_source_hash_mismatch(context, field):
    _, path, payload = _worker(context)
    if field == "artifact":
        payload["artifact"]["sha256"] = "0" * 64
    else:
        payload["source_sha256"]["one.py"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash|source"):
        context["parallel"].validate_worker_manifest(path, context["registry"])


def test_status_replace_keeps_old_value_until_complete_temp_exists(tmp_path, monkeypatch):
    parallel = _parallel()
    path = tmp_path / "status.json"
    path.write_text('{"state":"running"}\n')
    real_replace = os.replace
    observed = []

    def inspect_replace(source, target):
        observed.append((json.loads(path.read_text()), json.loads(source.read_text())))
        real_replace(source, target)

    monkeypatch.setattr(parallel.os, "replace", inspect_replace)
    parallel.write_status(path, {"state": "complete"})
    assert observed == [({"state": "running"}, {"state": "complete"})]
    assert json.loads(path.read_text()) == {"state": "complete"}


def test_promotion_requires_signed_paired_evaluation_and_coordinator(context, monkeypatch):
    parallel = context["parallel"]
    workers = [_worker(context, "0"), _worker(context, "1", artifact_bytes=b"fallback")]
    path, payload = _evaluation(context, workers)
    payload["signature"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="signature"):
        parallel.promote_artifact(path, "ring_expert")
    _sign(payload)
    path.write_text(json.dumps(payload))
    monkeypatch.delenv("V3_08_COORDINATOR_TOKEN")
    with pytest.raises(PermissionError, match="coordinator"):
        parallel.promote_artifact(path, "ring_expert")


def test_promotion_rejects_unranked_winner(context):
    workers = [_worker(context, "0"), _worker(context, "1", artifact_bytes=b"fallback")]
    path, payload = _evaluation(context, workers)
    payload["winner_run_id"], payload["fallback_run_id"] = payload["fallback_run_id"], payload["winner_run_id"]
    _sign(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rank|winner"):
        context["parallel"].promote_artifact(path, "ring_expert")


def test_promotion_publishes_winner_and_fallback_once(context):
    parallel = context["parallel"]
    workers = [_worker(context, "0"), _worker(context, "1", artifact_bytes=b"fallback")]
    path, _ = _evaluation(context, workers)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _attempt_promotion(parallel, path), range(2)))
    winners = [item for item in results if isinstance(item, Path)]
    assert len(winners) == 1
    assert sum(isinstance(item, FileExistsError) for item in results) == 1
    promoted = winners[0]
    assert promoted.read_bytes() == b"checkpoint-v1"
    assert (promoted.parent / "fallback.bin").read_bytes() == b"fallback"
    stage = json.loads((promoted.parent / "stage_manifest.json").read_text())
    assert stage["winner"]["run_id"] == workers[0][0].run_id
    assert stage["fallback"]["run_id"] == workers[1][0].run_id


def _attempt_promotion(parallel, path):
    try:
        return parallel.promote_artifact(path, "ring_expert")
    except Exception as exc:
        return exc
