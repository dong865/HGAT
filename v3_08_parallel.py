"""Frozen inputs, isolated workers and signed stage gates for V3 4v2."""

import argparse
from dataclasses import dataclass
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
OUTPUT_ROOT = WORKSPACE_ROOT / "checkpoints" / "v3_08_parallel"
SOURCE_ROOT = Path(__file__).resolve().parent
BASELINE_PATH = OUTPUT_ROOT / "baseline" / "dcap08c_best_4v2.pth"
SEED_REGISTRY_PATH = OUTPUT_ROOT / "coordination" / "seed_registry.json"
DATA_STAGES = frozenset({"feasibility", "ring_expert", "counterfactual"})
TRAINING_STAGES = frozenset({"actor", "allocation"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PROTOCOL = {"name": "deploy5x5", "d_cap": 0.8,
             "pursuer_speed": 0.35, "evader_speed": 0.35}
_COORDINATOR_ENV = "V3_08_COORDINATOR_TOKEN"
_EVALUATOR_ENV = "V3_08_EVALUATOR_KEY"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    directory: Path
    stage: str
    arm: str
    worker_id: str
    seeds: dict
    command: tuple[str, ...]


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _canonical_hash(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _interval(pair):
    if not isinstance(pair, list) or len(pair) != 2 or not all(type(n) is int for n in pair) or pair[0] > pair[1]:
        raise ValueError("invalid seed interval")
    return range(pair[0], pair[1] + 1)


def _check_allocations(registry):
    if registry.get("protocol") != _PROTOCOL:
        raise ValueError("protocol drift from frozen deploy5x5/d_cap=0.8 contract")
    if (registry.get("training_pool") != [72000, 72999]
            or registry.get("validation_pool") != [73000, 73199]
            or registry.get("generalization_pool") != [74000, 74199]
            or registry.get("sealed_seed") != 70086):
        raise ValueError("seed registry drift from frozen split")
    pools = [set(_interval(registry[key])) for key in
             ("training_pool", "validation_pool", "generalization_pool")]
    if pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2]:
        raise ValueError("seed pools overlap")
    if any(70086 in pool for pool in pools):
        raise ValueError("sealed seed enters a pool")
    stage_seeds = registry.get("stage_seeds", {})
    if any(type(stage_seeds.get(stage)) is not int for stage in DATA_STAGES | TRAINING_STAGES):
        raise ValueError("missing deterministic stage seeds")
    for stage, workers in registry["allocations"].items():
        if stage not in DATA_STAGES:
            raise ValueError("unknown allocation stage")
        seen = set()
        for worker, interval in workers.items():
            allocated = set(_interval(interval))
            if not allocated <= pools[0]:
                raise ValueError(f"{stage}/{worker} seeds outside training pool")
            if seen & allocated:
                raise ValueError(f"{stage} worker allocations overlap")
            seen.update(allocated)
    for stage, policy in registry.get("selection", {}).items():
        if stage not in DATA_STAGES | TRAINING_STAGES or policy.get("primary") not in ("full_capture_rate", "feasible_rate"):
            raise ValueError("invalid predeclared selection metric")
        if type(policy.get("maximize")) is not bool or policy.get("tie_break") != "mean_capture_time":
            raise ValueError("invalid predeclared ranking policy")


def _safe_name(value, label):
    name = str(value)
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid {label}: {name!r}")
    return name


def _source_files(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("source root is missing")
    files = {}
    for path in sorted(root.rglob("*")):
        if any(part in ("__pycache__", ".pytest_cache", ".git") for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"source tree contains symlink: {path}")
        if path.is_file() and path.suffix != ".pyc":
            files[path.relative_to(root).as_posix()] = _digest(path)
    if not files:
        raise ValueError("source tree is empty")
    return files


def write_status(path, status):
    """Publish complete JSON via a same-directory temporary file and rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(status, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _secret_hash(name):
    value = os.environ.get(name)
    if not value:
        raise PermissionError(f"{name} is required for coordinator/evaluator authorization")
    return hashlib.sha256(value.encode()).hexdigest()


def freeze_inputs(*, refresh=False):
    """Freeze every source file, the staged baseline, registry and gate keys."""
    source_root = Path(SOURCE_ROOT).resolve()
    baseline = Path(BASELINE_PATH).resolve()
    registry_path = Path(SEED_REGISTRY_PATH).resolve()
    if "ros2_ws_6V2" in baseline.parts:
        raise ValueError("baseline must be staged inside the isolated workspace")
    registry = _read_json(registry_path)
    _check_allocations(registry)
    files = _source_files(source_root)
    frozen = {
        "schema": 1,
        "source_root": str(source_root),
        "source_files": files,
        "source_tree_sha256": _canonical_hash(files),
        "baseline": {"path": str(baseline), "sha256": _digest(baseline)},
        "seed_registry": {"path": str(registry_path), "sha256": _digest(registry_path)},
        "protocol": registry["protocol"],
        "coordinator_token_sha256": _secret_hash(_COORDINATOR_ENV),
        "evaluator_key_sha256": _secret_hash(_EVALUATOR_ENV),
    }
    destination = Path(OUTPUT_ROOT) / "coordination" / "frozen_source.json"
    if destination.exists():
        previous = _read_json(destination)
        if previous == frozen:
            return previous
        if not refresh:
            raise ValueError("frozen inputs already exist and differ")
        if any(any((Path(OUTPUT_ROOT) / category).glob("**/config.json"))
               for category in ("datasets", "runs")):
            raise ValueError("cannot refresh frozen inputs after run reservation")
        if _secret_hash(_COORDINATOR_ENV) != previous["coordinator_token_sha256"]:
            raise PermissionError("coordinator authorization failed for refresh")
    write_status(destination, frozen)
    return frozen


def _load_frozen():
    frozen_path = Path(OUTPUT_ROOT) / "coordination" / "frozen_source.json"
    if not frozen_path.is_file():
        raise ValueError("frozen source gate is missing")
    frozen = _read_json(frozen_path)
    if frozen.get("schema") != 1:
        raise ValueError("unsupported frozen source schema")
    source_root = Path(SOURCE_ROOT).resolve()
    baseline = Path(BASELINE_PATH).resolve()
    registry_path = Path(SEED_REGISTRY_PATH).resolve()
    if (frozen.get("source_root") != str(source_root)
            or frozen.get("baseline", {}).get("path") != str(baseline)
            or frozen.get("seed_registry", {}).get("path") != str(registry_path)):
        raise ValueError("frozen input path mismatch")
    current_files = _source_files(source_root)
    if current_files != frozen.get("source_files") or _canonical_hash(current_files) != frozen.get("source_tree_sha256"):
        raise ValueError("frozen source tree hash mismatch")
    if _digest(baseline) != frozen["baseline"]["sha256"]:
        raise ValueError("frozen baseline hash mismatch")
    if _digest(registry_path) != frozen["seed_registry"]["sha256"]:
        raise ValueError("frozen registry hash mismatch")
    registry = _read_json(registry_path)
    _check_allocations(registry)
    if registry["protocol"] != frozen.get("protocol"):
        raise ValueError("frozen protocol mismatch")
    return frozen, registry


def _run_directory(stage, arm, worker_id, run_id):
    if stage in DATA_STAGES:
        return Path(OUTPUT_ROOT) / "datasets" / stage / f"shard_{worker_id}"
    return Path(OUTPUT_ROOT) / "runs" / stage / run_id


def _resource_assignment(stage, worker_id, registry):
    if not worker_id.isdecimal():
        raise ValueError("numeric worker_id is required for resource assignment")
    slot = int(worker_id)
    slots = len(registry["allocations"].get(stage, {})) if stage in DATA_STAGES else 6
    if slot >= slots:
        raise ValueError("worker_id exceeds worker budget")
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    if len(cpus) < slots:
        raise ValueError("insufficient CPUs for isolated worker affinity")
    chunks = [cpus[index::slots] for index in range(slots)]
    assigned = chunks[slot]
    return {"cpu_affinity": assigned, "omp_threads": len(assigned),
            "mkl_threads": len(assigned), "torch_threads": len(assigned),
            "cuda_visible_devices": "0" if stage in TRAINING_STAGES else "",
            "gpu_shared": True, "gpu_memory_fraction": 0.14 if stage in TRAINING_STAGES else 0.0,
            "max_workers": slots}


def _rng_seeds(stage_seed, worker_id, run_id):
    raw = hashlib.sha256(f"{stage_seed}:{worker_id}:{run_id}".encode()).digest()
    base = int.from_bytes(raw[:4], "big") % (2**31 - 5)
    return {name: base + offset for offset, name in enumerate(("python", "numpy", "torch", "environment"))}


def create_run(stage, arm, worker_id):
    """Reserve a run and return a launchable, resource-isolated command."""
    stage = _safe_name(stage, "stage")
    worker_id = _safe_name(worker_id, "worker_id")
    if stage not in DATA_STAGES | TRAINING_STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if not isinstance(arm, dict) or not isinstance(arm.get("argv"), list) or not arm["argv"]:
        raise ValueError("arm must declare name and executable argv")
    arm_name = _safe_name(arm.get("name"), "arm")
    if any(not isinstance(part, str) or not part for part in arm["argv"]):
        raise ValueError("arm argv must contain nonempty strings")
    frozen, registry = _load_frozen()
    if _secret_hash(_COORDINATOR_ENV) != frozen["coordinator_token_sha256"]:
        raise PermissionError("coordinator authorization failed")
    run_id = f"{stage}-{arm_name}-{worker_id}"
    if stage in DATA_STAGES:
        allocation = registry["allocations"].get(stage, {}).get(worker_id)
        if allocation is None:
            raise ValueError(f"no seed allocation for {stage}/{worker_id}")
        seeds = {"train": list(_interval(allocation)), "validation": []}
    else:
        seeds = {"train": list(_interval(registry["training_pool"])),
                 "validation": list(_interval(registry["validation_pool"]))}
    directory = _run_directory(stage, arm_name, worker_id, run_id)
    resources = _resource_assignment(stage, worker_id, registry)
    directory.mkdir(parents=True, exist_ok=False)
    temp_dir = directory / "tmp"
    temp_dir.mkdir()
    substitutions = {"{run_dir}": str(directory), "{config}": str(directory / "config.json"),
                     "{seeds}": str(directory / "seeds.json"), "{tmp_dir}": str(temp_dir)}
    worker_argv = []
    for part in arm["argv"]:
        for marker, value in substitutions.items():
            part = part.replace(marker, value)
        worker_argv.append(part)
    config = {"stage": stage, "arm": arm_name, "worker_id": worker_id, "run_id": run_id,
              "directory": str(directory), "temp_dir": str(temp_dir),
              "protocol": registry["protocol"], "worker_argv": worker_argv,
              "resources": resources, "rng_seeds": _rng_seeds(registry["stage_seeds"][stage], worker_id, run_id),
              "source_tree_sha256": frozen["source_tree_sha256"],
              "baseline_sha256": frozen["baseline"]["sha256"],
              "registry_sha256": frozen["seed_registry"]["sha256"]}
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "seeds.json").write_text(json.dumps(seeds, indent=2, sort_keys=True) + "\n")
    (directory / "source_sha256.json").write_text(json.dumps(frozen["source_files"], indent=2, sort_keys=True) + "\n")
    (directory / "worker.pid").write_text("0\n")
    (directory / "train.log").touch(exist_ok=False)
    (directory / "metrics.jsonl").touch(exist_ok=False)
    write_status(directory / "status.json", {"state": "created", "run_id": run_id})
    command = (sys.executable, str(Path(__file__).resolve()), "launch", "--config", str(directory / "config.json"))
    return RunSpec(run_id, directory, stage, arm_name, worker_id, seeds, command)


def _launch(config_path):
    """Run one reserved command with secrets stripped from the child environment."""
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    directory = Path(config["directory"]).resolve()
    if config_path != directory / "config.json" or not directory.is_dir():
        raise ValueError("launch config is outside reserved run directory")
    resources = config["resources"]
    environment = os.environ.copy()
    environment.pop(_COORDINATOR_ENV, None)
    environment.pop(_EVALUATOR_ENV, None)
    environment.update({"TMPDIR": config["temp_dir"], "OMP_NUM_THREADS": str(resources["omp_threads"]),
                        "MKL_NUM_THREADS": str(resources["mkl_threads"]),
                        "CUDA_VISIBLE_DEVICES": resources["cuda_visible_devices"],
                        "PYTHONHASHSEED": str(config["rng_seeds"]["python"]),
                        "V3_08_RUN_CONFIG": str(config_path),
                        "V3_08_GPU_MEMORY_FRACTION": str(resources["gpu_memory_fraction"])})
    affinity = resources["cpu_affinity"]
    preexec = (lambda: os.sched_setaffinity(0, set(affinity))) if hasattr(os, "sched_setaffinity") else None
    write_status(directory / "status.json", {"state": "running", "run_id": config["run_id"]})
    with (directory / "train.log").open("a") as log:
        child = subprocess.Popen(config["worker_argv"], cwd=directory, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, preexec_fn=preexec)
        (directory / "worker.pid").write_text(f"{child.pid}\n")
        code = child.wait()
    write_status(directory / "status.json", {"state": "complete" if code == 0 else "failed",
                                                  "run_id": config["run_id"], "exit_code": code})
    return code


def _contains_sealed(value, sealed):
    if isinstance(value, dict):
        return any(_contains_sealed(item, sealed) for item in value.values())
    if isinstance(value, list):
        return any(_contains_sealed(item, sealed) for item in value)
    if type(value) is int:
        return value == sealed
    if isinstance(value, str):
        return re.search(rf"(?<!\d){sealed}(?!\d)", value) is not None
    return False


def _contained_file(directory, name, label):
    if not isinstance(name, str) or not name:
        raise ValueError(f"missing {label} path")
    candidate = Path(name)
    if candidate.is_absolute():
        raise ValueError(f"{label} path must be relative to reserved run directory")
    path = (directory / candidate).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"{label} outside reserved run directory or missing")
    return path


def _verify_file(directory, entry, label):
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or not _HASH.fullmatch(entry["sha256"]):
        raise ValueError(f"invalid {label} hash")
    path = _contained_file(directory, entry.get("path"), label)
    if _digest(path) != entry["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return path


def validate_worker_manifest(path, seed_registry):
    """Bind one worker result to its reserved run and frozen inputs."""
    frozen, registry = _load_frozen()
    if Path(seed_registry).resolve() != Path(SEED_REGISTRY_PATH).resolve():
        raise ValueError("seed registry must be the frozen registry file")
    path = Path(path).resolve()
    manifest = _read_json(path)
    stage = _safe_name(manifest.get("stage"), "stage")
    arm = _safe_name(manifest.get("arm"), "arm")
    worker_id = _safe_name(manifest.get("worker_id"), "worker_id")
    run_id = f"{stage}-{arm}-{worker_id}"
    if stage not in DATA_STAGES | TRAINING_STAGES or manifest.get("run_id") != run_id:
        raise ValueError("manifest run identity mismatch")
    directory = _run_directory(stage, arm, worker_id, run_id).resolve()
    if path != directory / "artifact_manifest.json":
        raise ValueError("manifest outside reserved run directory")
    config_path = directory / "config.json"
    if not config_path.is_file() or not (directory / "seeds.json").is_file():
        raise ValueError("reserved run config or seeds missing")
    config = _read_json(config_path)
    expected_config = {"stage": stage, "arm": arm, "worker_id": worker_id,
                       "run_id": run_id, "directory": str(directory),
                       "source_tree_sha256": frozen["source_tree_sha256"],
                       "baseline_sha256": frozen["baseline"]["sha256"],
                       "registry_sha256": frozen["seed_registry"]["sha256"],
                       "protocol": registry["protocol"]}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("reserved run config mismatch")
    seeds = _read_json(directory / "seeds.json")
    if manifest.get("seeds") != seeds:
        raise ValueError("manifest seeds differ from reserved seeds")
    train = seeds.get("train", [])
    validation = seeds.get("validation", [])
    if (not isinstance(train, list) or not isinstance(validation, list)
            or any(type(seed) is not int for seed in train + validation)
            or len(train) != len(set(train)) or len(validation) != len(set(validation))
            or set(train) & set(validation)):
        raise ValueError("invalid or duplicate worker seeds")
    if stage in DATA_STAGES:
        allocation = registry["allocations"].get(stage, {}).get(worker_id)
        if allocation is None or train != list(_interval(allocation)) or validation:
            raise ValueError("worker seeds outside allocation")
    elif train != list(_interval(registry["training_pool"])) or validation != list(_interval(registry["validation_pool"])):
        raise ValueError("training arm seeds differ from paired frozen split")
    if manifest.get("status") != "complete" or _read_json(directory / "status.json").get("state") != "complete":
        raise ValueError("worker artifact is not complete")
    if (_contains_sealed(manifest, registry["sealed_seed"])
            or manifest.get("source_sha256") != frozen["source_files"]
            or manifest.get("source_tree_sha256") != frozen["source_tree_sha256"]
            or manifest.get("baseline_sha256") != frozen["baseline"]["sha256"]
            or manifest.get("registry_sha256") != frozen["seed_registry"]["sha256"]):
        if _contains_sealed(manifest, registry["sealed_seed"]):
            raise ValueError("sealed seed or derived scene in worker manifest")
        raise ValueError("worker source/baseline/registry hash mismatch")
    if _read_json(directory / "source_sha256.json") != frozen["source_files"]:
        raise ValueError("reserved source hash file mismatch")
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("invalid sample records")
    if stage in DATA_STAGES:
        sample_path = _verify_file(directory, manifest.get("sample_manifest"), "sample manifest")
        sample_doc = _read_json(sample_path)
        if _contains_sealed(sample_doc, registry["sealed_seed"]):
            raise ValueError("sealed sample content")
        if sample_doc.get("samples") != samples:
            raise ValueError("sample content manifest differs from worker manifest")
    for sample in samples:
        if not isinstance(sample, dict) or type(sample.get("scenario_seed")) is not int or sample["scenario_seed"] not in train:
            raise ValueError("sample seed outside worker allocation")
        if type(sample.get("origin_seed")) is not int or sample["origin_seed"] not in train:
            raise ValueError("sample origin seed outside worker allocation")
    _verify_file(directory, manifest.get("artifact"), "artifact")
    result = dict(manifest)
    result["_manifest_path"] = str(path)
    return result


def _rank_key(candidate, policy):
    metrics = candidate.get("metrics", {})
    primary = metrics.get(policy["primary"])
    tie_break = metrics.get(policy["tie_break"])
    if (type(primary) not in (int, float) or type(tie_break) not in (int, float)
            or not math.isfinite(primary) or not math.isfinite(tie_break)):
        raise ValueError("nonfinite or missing predeclared evaluation metric")
    return ((-primary if policy["maximize"] else primary), tie_break, candidate["run_id"])


def promote_artifact(manifest, stage):
    """Publish the signed, ranked winner and fallback through one atomic rename."""
    frozen, registry = _load_frozen()
    stage = _safe_name(stage, "stage")
    if _secret_hash(_COORDINATOR_ENV) != frozen["coordinator_token_sha256"]:
        raise PermissionError("coordinator authorization failed")
    key = os.environ.get(_EVALUATOR_ENV)
    if not key or hashlib.sha256(key.encode()).hexdigest() != frozen["evaluator_key_sha256"]:
        raise PermissionError("evaluator key is unavailable")
    path = Path(manifest).resolve()
    evaluation_root = (Path(OUTPUT_ROOT) / "evaluation" / stage).resolve()
    if not path.is_relative_to(evaluation_root) or not path.is_file():
        raise ValueError("evaluation result is outside stage evaluation directory")
    result = _read_json(path)
    signature = result.get("signature")
    unsigned = {name: value for name, value in result.items() if name != "signature"}
    expected_signature = hmac.new(key.encode(), _canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(signature, str) or not hmac.compare_digest(signature, expected_signature):
        raise ValueError("evaluation signature mismatch")
    if result.get("stage") != stage or result.get("protocol") != registry["protocol"] or result.get("source_tree_sha256") != frozen["source_tree_sha256"]:
        raise ValueError("evaluation protocol/source mismatch")
    evaluation_seeds = list(_interval(registry["generalization_pool"]))
    if result.get("evaluation_seeds") != evaluation_seeds:
        raise ValueError("evaluation did not use frozen generalization seeds")
    baseline = result.get("baseline", {})
    if baseline.get("checkpoint_sha256") != frozen["baseline"]["sha256"] or baseline.get("seeds") != evaluation_seeds:
        raise ValueError("paired baseline evaluation mismatch")
    policy = registry.get("selection", {}).get(stage)
    if policy is None:
        raise ValueError("no predeclared selection policy for stage")
    _rank_key({"run_id": "baseline", "metrics": baseline.get("metrics")}, policy)
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("winner and fallback require two evaluated candidates")
    validated = {}
    for candidate in candidates:
        if candidate.get("seeds") != evaluation_seeds or candidate.get("source_tree_sha256") != frozen["source_tree_sha256"]:
            raise ValueError("candidate evaluation was not paired on frozen seeds/source")
        worker = validate_worker_manifest(candidate.get("manifest_path"), SEED_REGISTRY_PATH)
        if (worker["stage"] != stage or worker["run_id"] != candidate.get("run_id")
                or worker["artifact"]["sha256"] != candidate.get("checkpoint_sha256")):
            raise ValueError("candidate checkpoint or worker identity mismatch")
        if worker["run_id"] in validated:
            raise ValueError("duplicate evaluated run")
        _rank_key(candidate, policy)
        validated[worker["run_id"]] = worker
    ranked = sorted(candidates, key=lambda item: _rank_key(item, policy))
    if (result.get("winner_run_id") != ranked[0]["run_id"]
            or result.get("fallback_run_id") != ranked[1]["run_id"]):
        raise ValueError("winner/fallback ranking violates predeclared metrics")
    promoted_root = Path(OUTPUT_ROOT) / "promoted"
    promoted_root.mkdir(parents=True, exist_ok=True)
    destination = promoted_root / stage
    if destination.exists():
        raise FileExistsError(f"stage already promoted: {stage}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{stage}-", dir=promoted_root))
    try:
        selected = {}
        for label, candidate in (("winner", ranked[0]), ("fallback", ranked[1])):
            worker = validated[candidate["run_id"]]
            source = _contained_file(Path(worker["_manifest_path"]).parent, worker["artifact"]["path"], "artifact")
            copied = temporary / f"{label}{source.suffix}"
            shutil.copy2(source, copied)
            if _digest(copied) != worker["artifact"]["sha256"]:
                raise ValueError("promoted artifact hash mismatch")
            copied.chmod(0o444)
            selected[label] = {"run_id": worker["run_id"], "path": copied.name,
                               "sha256": worker["artifact"]["sha256"]}
        stage_manifest = {"stage": stage, "evaluation_sha256": _digest(path),
                          "source_tree_sha256": frozen["source_tree_sha256"],
                          "baseline_sha256": frozen["baseline"]["sha256"], **selected}
        stage_path = temporary / "stage_manifest.json"
        stage_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n")
        stage_path.chmod(0o444)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(f"stage already promoted: {stage}") from exc
            raise
        return destination / selected["winner"]["path"]
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch"])
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    return _launch(arguments.config)


if __name__ == "__main__":
    sys.exit(_main())
