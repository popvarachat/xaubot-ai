"""GOLDmicro model registry primitives.

This module manages metadata and paths only. It never places orders, never changes
MT5 settings, and never activates a candidate. Binary model artifacts are local
and versioned by manifest/hash; GitHub stores auditable manifests/reports.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ModelPaths:
    root: Path
    active: Path
    candidates: Path
    archive: Path
    manifests: Path
    reports: Path


def lifecycle_paths(root: str | Path = "models") -> ModelPaths:
    root = Path(root)
    return ModelPaths(
        root=root,
        active=root / "active",
        candidates=root / "candidates",
        archive=root / "archive",
        manifests=root / "manifests",
        reports=root / "reports",
    )


def ensure_lifecycle_dirs(root: str | Path = "models") -> ModelPaths:
    paths = lifecycle_paths(root)
    for p in (paths.active, paths.candidates, paths.archive, paths.manifests, paths.reports):
        p.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: str | Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelManifest:
    model_id: str
    status: str
    git_sha: str
    created_at: str
    training_start: str
    training_end: str
    feature_set: str
    config_hash: str
    random_seed: int
    xgb_path: str
    hmm_path: str
    xgb_sha256: str = ""
    hmm_sha256: str = ""
    validation_report: str = ""
    shadow_report: str = ""
    rollback_model_id: str = ""
    approval_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(manifest: ModelManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> ModelManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelManifest(**data)


def candidate_dir(model_id: str, root: str | Path = "models") -> Path:
    if not model_id or any(ch in model_id for ch in ("/", "\\", "..")):
        raise ValueError("unsafe model_id")
    return lifecycle_paths(root).candidates / model_id


def active_manifest_path(root: str | Path = "models") -> Path:
    return lifecycle_paths(root).active / "champion_manifest.json"


def build_activation_plan(
    candidate: ModelManifest,
    *,
    current_champion_id: str,
    human_approved: bool,
    positions_open: int,
    closed_candle_boundary: bool,
) -> dict[str, Any]:
    """Return an auditable plan. This function intentionally performs no activation."""
    blockers: list[str] = []
    if candidate.status != "ELIGIBLE_FOR_HUMAN_GATE":
        blockers.append("candidate is not promotion-eligible")
    if not human_approved:
        blockers.append("Human Gate approval missing")
    if positions_open:
        blockers.append("positions are still open")
    if not closed_candle_boundary:
        blockers.append("activation must wait for next newly closed execution candle")
    if not candidate.rollback_model_id:
        blockers.append("rollback_model_id missing")
    if candidate.rollback_model_id and candidate.rollback_model_id != current_champion_id:
        blockers.append("rollback reference does not match current Champion")

    return {
        "candidate_model_id": candidate.model_id,
        "current_champion_id": current_champion_id,
        "ready": not blockers,
        "blockers": blockers,
        "action": "HUMAN_GATED_ATOMIC_ACTIVATION" if not blockers else "NO_CHANGE",
    }
