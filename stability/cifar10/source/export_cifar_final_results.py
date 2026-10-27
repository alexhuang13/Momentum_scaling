#!/usr/bin/env python3
"""Export saved CIFAR stability results for a reproducible combined figure.

Uses only the Python standard library. It reads existing results and writes a
new ZIP; it neither imports the training program nor starts any GPU work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile


TARGETED = {
    "01_sgd_B1_transition": 40,
    "02_polyak_B32_transition": 35,
    "03_polyak_B128_transition": 24,
    "04_polyak_B256_transition": 18,
    "05_polyak_B4_gap0125": 35,
    "06a_nesterov_B4_gap_1over128": 30,
    "06b_nesterov_B4_gap_1over64_safety": 2,
    "06c_nesterov_B4_gap_1over32_safety": 2,
    "06d_nesterov_B4_gap_1over16_safety": 8,
    "06e_nesterov_B4_gap_1over8_safety": 6,
}
OLD_RUNS = ("fine_grid_seed0_2gpu", "refine_5seed_2gpu")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_history(path):
    """Retain every probe, initial/final rows, stops, and finite metric peaks."""
    keep, peaks, last = {}, {}, None
    metrics = ("train_loss", "gradient_norm", "parameter_norm")
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                step = int(row["step"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Malformed history: {path}, line {number}") from exc
            if last is not None and step <= int(last["step"]):
                raise ValueError(f"History steps are not increasing: {path}")
            if step == 0 or "probe_loss" in row or row.get("stop_reason"):
                keep[step] = row
            for metric in metrics:
                value = row.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    if metric not in peaks or value > peaks[metric][metric]:
                        peaks[metric] = row
            last = row
    if last is None:
        raise ValueError(f"Empty history: {path}")
    keep[int(last["step"])] = last
    for row in peaks.values():
        keep[int(row["step"])] = row
    return "".join(json.dumps(keep[step]) + "\n" for step in sorted(keep))


def export(project, destination=None):
    project = project.expanduser().resolve()
    targeted_root = project / "output" / "targeted_reruns_20260920"
    roots = [project / "output" / name for name in OLD_RUNS]
    roots += [targeted_root / name for name in TARGETED]
    missing = [str(root / "manifest.json") for root in roots
               if not (root / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing result manifests:\n" + "\n".join(missing))

    manifests = {}
    for root in roots:
        manifest = read_json(root / "manifest.json")
        if not isinstance(manifest.get("jobs"), list):
            raise ValueError(f"Invalid manifest: {root}")
        body = {k: v for k, v in manifest.items() if k != "fingerprint"}
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(f"Manifest checksum mismatch: {root}")
        for trial in manifest["jobs"]:
            trial_id = trial["id"]
            if not isinstance(trial_id, str) or len(trial_id) != 20 or any(
                    c not in "0123456789abcdef" for c in trial_id):
                raise ValueError(f"Invalid trial id in {root}")
        manifests[root] = manifest

    destination = (destination or project / "output" / "summary_exports").expanduser().resolve()
    # Keep the new export outside all input run directories.
    if any(destination == root or root in destination.parents for root in roots):
        raise ValueError("Choose an export directory outside the input run directories")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    archive = destination / f"cifar_final_results_{stamp}.zip"
    status = []
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("cifar_stability.py", "cifar_stability_revised.py",
                     "rerun_cifar_outliers_minimal_output.sh"):
            path = project / name
            if path.is_file():
                bundle.write(path, f"code/{name}")
        for root in roots:
            manifest = manifests[root]
            prefix = root.relative_to(project).as_posix()
            for name in ("manifest.json", "source_manifest.json", "refinement_plan.csv"):
                path = root / name
                if path.is_file():
                    bundle.write(path, f"{prefix}/{name}")
            entry = dict(source=prefix, planned=len(manifest["jobs"]),
                         expected=TARGETED.get(root.name), saved_results=0,
                         exported_histories=0, missing_results=[], missing_histories=[],
                         steps=manifest["config"]["steps"],
                         tail_evals=manifest["config"]["tail_evals"],
                         parameter_safety_mode=manifest["config"].get(
                             "parameter_safety_mode", "stop"))
            for trial in manifest["jobs"]:
                folder = root / "trials" / trial["id"]
                summary_path = folder / "summary.json"
                if not summary_path.is_file():
                    entry["missing_results"].append(trial["id"])
                    continue
                summary = read_json(summary_path)
                if summary.get("trial") != trial:
                    raise ValueError(f"Trial/summary mismatch: {summary_path}")
                imported = manifest.get("reused_results", {}).get(trial["id"], {})
                if summary.get("fingerprint") != imported.get(
                        "fingerprint", manifest["fingerprint"]):
                    raise ValueError(f"Summary fingerprint mismatch: {summary_path}")
                if imported.get("summary_digest") and hashlib.sha256(
                        json.dumps(summary, sort_keys=True).encode()).hexdigest() != imported["summary_digest"]:
                    raise ValueError(f"Reused summary checksum mismatch: {summary_path}")
                trial_prefix = f"{prefix}/trials/{trial['id']}"
                bundle.write(summary_path, f"{trial_prefix}/summary.json")
                if (folder / "config.json").is_file():
                    bundle.write(folder / "config.json", f"{trial_prefix}/config.json")
                entry["saved_results"] += 1
                history_path = folder / "history.jsonl"
                if not history_path.is_file():
                    entry["missing_histories"].append(trial["id"])
                    continue
                bundle.writestr(f"{trial_prefix}/diagnostic_history.jsonl",
                                compact_history(history_path))
                entry["exported_histories"] += 1
            status.append(entry)
            print(f"{root.name}: {entry['saved_results']}/{entry['planned']} saved; "
                  f"{entry['exported_histories']} histories", flush=True)
        bundle.writestr("export_status.json", json.dumps(status, indent=2) + "\n")
        bundle.writestr("README.txt",
            "Original manifests, trial configs and summaries are preserved unchanged.\n"
            "diagnostic_history.jsonl retains every probe evaluation, initialization,\n"
            "stop events, the final row, and finite loss/gradient/parameter norm peaks.\n"
            "It is a subset of history.jsonl; intervening training-only rows are omitted.\n"
            "Saved results include trials terminated by safety guards.\n"
            "The old and revised stability definitions differ; do not pool bounded labels.\n"
            "Recompute compatible completed histories using tail_evals=10.\n"
            "Old parameter-only stops are censored under record mode, unless replaced\n"
            "by a corresponding revised run. Preserve provenance and seed counts.\n"
            "The gap=1/128 Nesterov experiment uses K=4000, not K=2000.\n")
    new = [entry for entry in status if entry["expected"] is not None]
    total = sum(entry["saved_results"] for entry in new)
    planned = sum(entry["planned"] for entry in new)
    print(f"\nTargeted reruns: {total}/{planned} saved (script plan: 200).")
    if any(entry["expected"] != entry["planned"] for entry in new):
        print("NOTE: at least one saved manifest differs from the original 200-trial plan.")
    if any(entry["missing_results"] for entry in status):
        print("NOTE: missing results are listed in export_status.json; none were invented.")
    if any(entry["missing_histories"] for entry in status):
        print("NOTE: missing histories are listed; some reused trials may have them in the old scan.")
    print(f"\n请上传这个文件：\n{archive}")
    print(f"大小：{archive.stat().st_size / 1024**2:.2f} MiB")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path("/home/jnwang/momentum/cifar10_stability"))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    export(args.project, args.destination)


if __name__ == "__main__":
    main()
