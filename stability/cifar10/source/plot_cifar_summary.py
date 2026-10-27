#!/usr/bin/env python3
"""Merge CIFAR scans and reruns; count unreplaced old parameter stops as unstable.

python3 plot_cifar_summary.py --input cifar_final_results_*.zip --output new_summary
The input may instead be the project/extracted directory containing output/.
Only NumPy and Matplotlib are needed; this file never starts training.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


OLD_HASH = "eef1a8b500f88bb66cf05b92ae570ddb0ac5ea59fb5985bfdaf9b0b5d94760cf"
NEW_HASH = "388b650f262371ccfb627e5593bfcdbda9e4a66a813b9b0f6b0fe30e724cb7ca"
LEGACY_HASH = "3be6106d28077c184cc9791f6ee12d5f89d8f3f810f732a4908bfd85ab24d852"
ALGORITHMS = ("sgd", "polyak", "nesterov")
TARGETED = {
    "01_sgd_B1_transition": 40, "02_polyak_B32_transition": 35,
    "03_polyak_B128_transition": 24, "04_polyak_B256_transition": 18,
    "05_polyak_B4_gap0125": 35, "06a_nesterov_B4_gap_1over128": 30,
    "06b_nesterov_B4_gap_1over64_safety": 2,
    "06c_nesterov_B4_gap_1over32_safety": 2,
    "06d_nesterov_B4_gap_1over16_safety": 8,
    "06e_nesterov_B4_gap_1over8_safety": 6,
}
SOURCES = ["output/fine_grid_seed0_2gpu", "output/refine_5seed_2gpu"] + [
    "output/targeted_reruns_20260920/" + name for name in TARGETED]
LR_TOL = 1e-6
FAILURE_REASONS = {"gradient_safety_limit", "probe_loss_safety_limit",
                   "nonfinite_training_loss_or_gradient", "nonfinite_parameters",
                   "nonfinite_probe_loss"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows, fieldnames=None):
    keys = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class Input:
    def __init__(self, path):
        self.path = path.resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive and len(self.archive.namelist()) != len(set(self.archive.namelist())):
            raise ValueError("Duplicate paths in the input ZIP")

    def read(self, name):
        return (self.archive.read(name) if self.archive else (self.path / name).read_bytes())

    def exists(self, name):
        return name in self.archive.namelist() if self.archive else (self.path / name).is_file()

    def json(self, name):
        return json.loads(self.read(name))

    def close(self):
        if self.archive:
            self.archive.close()


def read_history(bundle, prefix):
    for name in ("diagnostic_history.jsonl", "history.jsonl"):
        path = prefix + "/" + name
        if bundle.exists(path):
            rows = [json.loads(line) for line in bundle.read(path).splitlines() if line.strip()]
            if not rows or rows[0].get("step") != 0:
                raise ValueError("Missing initialization: " + path)
            steps = [row["step"] for row in rows]
            if steps != sorted(set(steps)):
                raise ValueError("Invalid history ordering: " + path)
            return rows
    raise ValueError("Missing history: " + prefix)


def ratios(probes, initial, window):
    if len(probes) < 2 * window:
        return None, None
    tail = statistics.fmean(probes[-window:])
    previous = statistics.fmean(probes[-2 * window:-window])
    return tail / initial, tail / max(previous, 1e-30)


def assess(summary, history, config, window=10):
    """Use the requested summary convention: old parameter stops count as failures."""
    reason = summary["stop_reason"]
    initial = summary["initial_probe_loss"]
    if history[0]["probe_loss"] != initial:
        raise ValueError("Initial probe loss mismatch")
    if reason == "parameter_safety_limit":
        return "unstable", "legacy_parameter_stop_counted_unstable", None, None
    if reason in FAILURE_REASONS:
        return "unstable", reason, None, None
    if reason:
        raise ValueError("Unrecognized stop reason: " + reason)
    if summary["updates_completed"] != config["steps"]:
        return "unknown", "unfinished", None, None
    probes = [r for r in history if r["step"] > 0 and "probe_loss" in r]
    expected_steps = list(range(config["eval_every"], config["steps"] + 1, config["eval_every"]))
    if config["steps"] % config["eval_every"]:
        expected_steps.append(config["steps"])
    if [r["step"] for r in probes] != expected_steps:
        raise ValueError("Incomplete probe history for a completed trial")
    losses = [r["probe_loss"] for r in probes]
    if any(v is None or not math.isfinite(v) for v in losses):
        raise ValueError("Nonfinite loss in an ostensibly completed trial")
    original_n = config["tail_evals"]
    if losses[-original_n:] != summary["tail_probe_losses"] or \
            losses[-2 * original_n:-original_n] != summary["previous_probe_losses"]:
        raise ValueError("Saved tail statistics do not match probe history")
    old_ratio, old_growth = ratios(losses, initial, original_n)
    old_pass = old_ratio <= config["stable_loss_ratio"] and old_growth <= config["growth_ratio"]
    if old_pass != summary["bounded"]:
        raise ValueError("Original bounded label does not match original criterion")
    ratio, growth = ratios(losses, initial, window)
    if ratio is None:
        return "unknown", "insufficient_probes", None, None
    causes = []
    if ratio > config["stable_loss_ratio"]:
        causes.append("tail_loss")
    if growth > config["growth_ratio"]:
        causes.append("tail_growth")
    return ("unstable" if causes else "stable"), ("+".join(causes) or "passed"), ratio, growth


def protocol(config):
    allowed = {"preset", "axis", "algorithms", "batches", "gaps", "fixed_batch",
               "fixed_gap", "lrs", "lr_min", "lr_max", "lr_count", "seeds", "device",
               "devices", "required_fraction", "steps", "tail_evals", "parameter_safety_mode"}
    return {key: value for key, value in config.items() if key not in allowed}


def audit_source_code(bundle):
    names = {"import_training_stack", "seed_everything", "group_norm_factory", "build_model",
             "build_transforms", "PaperMomentum", "optimizer_for", "cifar_integrity_metadata",
             "build_data", "tensor_norm", "evaluate", "backward_batch", "classify", "run_trial"}
    out = {}
    for filename, expected in (("cifar_stability.py", OLD_HASH), ("cifar_stability_revised.py", NEW_HASH)):
        path = "code/" + filename
        if not bundle.exists(path):
            # In a live project, code is at the project root instead of code/.
            path = filename
        text = bundle.read(path).decode()
        lines = text.splitlines()
        selected = {n.name: "\n".join(lines[n.lineno - 1:n.end_lineno])
                    for n in ast.parse(text).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
        code_hash = digest(selected)
        if len(selected) != len(names) or code_hash != expected:
            raise ValueError("Unaudited training code: " + filename)
        out[filename] = code_hash
    return out


def load_records(bundle):
    code_hashes = audit_source_code(bundle)
    records, reports, point_views = [], [], set()
    common = None
    data_ids, probe_ids, environments = set(), set(), set()
    for source_index, source in enumerate(SOURCES):
        manifest = bundle.json(source + "/manifest.json")
        if manifest["schema_version"] != 2 or digest({k: v for k, v in manifest.items()
                                                    if k != "fingerprint"}) != manifest["fingerprint"]:
            raise ValueError("Manifest checksum/schema error: " + source)
        config = manifest["config"]
        if config["fake_data"]:
            raise ValueError("FakeData is not scientific input")
        if common is None:
            common = protocol(config)
        if protocol(config) != common:
            raise ValueError("Unapproved protocol difference: " + source)
        mode = config.get("parameter_safety_mode", "stop")
        steps = config["steps"]
        if steps not in (2000, 4000) or config["tail_evals"] not in (5, 10):
            raise ValueError("Unexpected horizon/tail window")
        if steps == 4000 and Path(source).name != "06a_nesterov_B4_gap_1over128":
            raise ValueError("Unexpected K=4000 run")
        code_hash = manifest.get("training_code_hash")
        if mode not in ("stop", "record"):
            raise ValueError("Unrecognized parameter guard mode")
        if code_hash is None:
            if mode != "stop" or set(manifest["source_hashes"].values()) != {LEGACY_HASH}:
                raise ValueError("Unknown legacy training code")
        elif code_hash != (OLD_HASH if mode == "stop" else NEW_HASH):
            raise ValueError("Training code/protocol mismatch")
        expected = TARGETED.get(Path(source).name)
        if expected is not None and len(manifest["jobs"]) != expected:
            raise ValueError("Targeted grid does not match the 200-trial plan")
        for algorithm in manifest["algorithms"]:
            for p in manifest["points"]:
                gap = 1.0 if algorithm == "sgd" else p["gap"]
                point_views.add((p["axis"], algorithm, p["x"], p["batch"], gap, steps))
        for t in manifest["jobs"]:
            prefix = source + "/trials/" + t["id"]
            result = bundle.json(prefix + "/summary.json")
            imported = manifest.get("reused_results", {}).get(t["id"], {})
            if result["trial"] != t or result["fingerprint"] != imported.get(
                    "fingerprint", manifest["fingerprint"]):
                raise ValueError("Trial identity mismatch: " + prefix)
            if imported.get("summary_digest") and digest(result) != imported["summary_digest"]:
                raise ValueError("Reused summary checksum mismatch: " + prefix)
            if result.get("smoke_test") or result.get("fake_data"):
                raise ValueError("FakeData result detected")
            history = read_history(bundle, prefix)
            state, why, loss, growth = assess(result, history, config)
            data_ids.add(result["data"]["file_listing_hash"])
            probe_ids.add(digest(result["data"]["probe_indices"]))
            env = result["environment"]
            environments.add(tuple(env[k] for k in ("torch", "torchvision", "dtype", "deterministic_algorithms")))
            if t["algorithm"] not in ALGORITHMS or not math.isfinite(t["lr"]) or t["lr"] <= 0:
                raise ValueError("Invalid trial coordinate")
            records.append({
                "source": source, "source_index": source_index, "trial_id": t["id"],
                "algorithm": t["algorithm"], "batch": t["batch"],
                "gap": 1.0 if t["algorithm"] == "sgd" else t["gap"],
                "raw_lr": t["lr"], "seed": t["seed"], "K": steps,
                "parameter_safety_mode": mode, "source_tail_evals": config["tail_evals"],
                "updates_completed": result["updates_completed"], "stop_reason": result["stop_reason"],
                "original_bounded": result["bounded"], "state": state, "criterion_reason": why,
                "tail10_loss_ratio": loss, "tail10_growth_ratio": growth,
                "initial_probe_loss": result["initial_probe_loss"],
                "initial_parameter_norm": result["initial_parameter_norm"],
                "summary_digest": digest(result), "selected": False,
            })
        reports.append({"source": source, "planned": len(manifest["jobs"]),
                        "saved": len(manifest["jobs"]), "K": steps, "parameter_mode": mode,
                        "original_bounded": sum(r["original_bounded"] for r in records if r["source"] == source),
                        "fingerprint": manifest["fingerprint"]})
    if any(len(values) != 1 for values in (data_ids, probe_ids, environments)):
        raise ValueError("Dataset, fixed probe set, or numerical environment changed")
    return records, sorted(point_views), {
        "sources": reports, "common_protocol": common, "training_code_hashes": code_hashes,
        "dataset_hash": next(iter(data_ids)), "probe_indices_hash": next(iter(probe_ids)),
        "environment": list(next(iter(environments))),
    }


def merge_records(records):
    """Group only documented LR roundoff; choose record-mode rerun regardless of outcome."""
    values = sorted({r["raw_lr"] for r in records})
    clusters = []
    for value in values:
        if clusters and math.isclose(value, clusters[-1][0], rel_tol=LR_TOL, abs_tol=0):
            clusters[-1].append(value)
        else:
            clusters.append([value])
    mapping = {v: cluster[0] for cluster in clusters for v in cluster}
    grouped = defaultdict(list)
    for row in records:
        row["lr"] = mapping[row["raw_lr"]]
        key = tuple(row[k] for k in ("algorithm", "batch", "gap", "lr", "seed", "K"))
        grouped[key].append(row)
    selected, replacements, copies = [], [], 0
    for key, candidates in sorted(grouped.items()):
        best = sorted(candidates, key=lambda r: (r["parameter_safety_mode"] != "record", r["source_index"]))[0]
        for row in candidates:
            if row is best:
                continue
            if row["parameter_safety_mode"] == best["parameter_safety_mode"]:
                if row["summary_digest"] != best["summary_digest"]:
                    raise ValueError("Conflicting duplicate with the same protocol: " + str(key))
                copies += 1
            else:
                if row["initial_probe_loss"] != best["initial_probe_loss"] or \
                        row["initial_parameter_norm"] != best["initial_parameter_norm"]:
                    raise ValueError("Initialization mismatch between old and revised runs")
                replacements.append({"algorithm": key[0], "batch": key[1], "gap": key[2],
                    "lr": key[3], "seed": key[4], "K": key[5],
                    "old_source": row["source"], "old_trial": row["trial_id"],
                    "new_source": best["source"], "new_trial": best["trial_id"],
                    "old_lr": row["raw_lr"], "new_lr": best["raw_lr"],
                    "exact_lr_match": row["raw_lr"] == best["raw_lr"],
                    "old_state_under_new_rule": row["state"], "new_state": best["state"],
                    "old_stop_reason": row["stop_reason"], "new_stop_reason": best["stop_reason"]})
        best["selected"] = True
        selected.append(best)
    return selected, replacements, {
        "input_summaries": len(records), "unique_selected_trials": len(selected),
        "duplicates_or_superseded_removed": len(records) - len(selected),
        "unchanged_duplicate_copies": copies,
        "replaced_source_records": len(replacements),
        "lr_roundoff_groups": [group for group in clusters if len(group) > 1],
        "lr_relative_tolerance": LR_TOL,
        "selected_state_counts": {state: sum(r["state"] == state for r in selected)
                                  for state in ("stable", "unstable", "unknown")},
        "selected_unknown_reasons": dict(Counter(r["criterion_reason"] for r in selected if r["state"] == "unknown")),
    }


def build_cells(selected, views):
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["algorithm"], row["batch"], row["gap"], row["K"])].append(row)
    cells = []
    for axis, algorithm, x, batch, gap, steps in views:
        lr_rows = defaultdict(list)
        for row in grouped[(algorithm, batch, gap, steps)]:
            lr_rows[row["lr"]].append(row)
        for lr, rows in sorted(lr_rows.items()):
            stable = sum(r["state"] == "stable" for r in rows)
            unstable = sum(r["state"] == "unstable" for r in rows)
            unknown = len(rows) - stable - unstable
            valid = stable + unstable
            cells.append({"axis": axis, "algorithm": algorithm, "x": x, "batch": batch,
                "gap": gap, "K": steps, "lr": lr, "n_total": len(rows),
                "n_stable": stable, "n_unstable": unstable, "n_unknown": unknown,
                "n_assessed": valid, "stable_fraction_assessed": stable / valid if valid else None,
                "fraction_lower_bound": stable / len(rows),
                "fraction_upper_bound": (stable + unknown) / len(rows),
                "n_revised": sum(r["parameter_safety_mode"] == "record" for r in rows),
                "n_legacy_parameter_stop_failures": sum(r["stop_reason"] == "parameter_safety_limit" for r in rows),
                "seed_ids": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
                "unknown_seed_ids": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"]) if r["state"] == "unknown")})
    return cells


def summarize_changes(records, selected, replacements):
    # Compare each old physical trial once, even if copied into the refinement.
    old = {}
    for row in records:
        if row["parameter_safety_mode"] == "stop":
            old.setdefault((row["algorithm"], row["batch"], row["gap"], row["lr"], row["seed"], row["K"]), row)
    flips = [dict(row) for row in old.values() if row["state"] != "unknown"
             and row["original_bounded"] != (row["state"] == "stable")]
    unique_replacements = {tuple(r[k] for k in ("algorithm", "batch", "gap", "lr", "seed", "K")): r
                           for r in replacements}
    replaced_stops = [r for r in unique_replacements.values() if r["old_stop_reason"] == "parameter_safety_limit"]
    return flips, {
        "old_unique_trials": len(old), "revised_trials": sum(r["parameter_safety_mode"] == "record" for r in selected),
        "revised_original_bounded": sum(r["parameter_safety_mode"] == "record" and r["original_bounded"] for r in selected),
        "unique_old_trials_superseded": len(unique_replacements),
        "old_parameter_stops_rerun": len(replaced_stops),
        "rerun_old_parameter_stop_outcomes": dict(Counter(r["new_state"] for r in replaced_stops)),
        "window_reclassification_count": len(flips),
        "window_reclassification": dict(Counter(str(r["original_bounded"]) + " -> " + r["state"] for r in flips)),
    }


def log_edges(values):
    logs = [math.log(value) for value in values]
    if len(logs) == 1:
        return [math.exp(logs[0] - math.log(2)/4), math.exp(logs[0] + math.log(2)/4)]
    return [math.exp(logs[0] - (logs[1] - logs[0])/2)] + [
        math.exp((a+b)/2) for a, b in zip(logs, logs[1:])] + [math.exp(logs[-1] + (logs[-1]-logs[-2])/2)]


def write_notes(output, report, cells, selected):
    focus = [
        ("batch", "sgd", 1, 1.189207), ("batch", "sgd", 1, 1.414214),
        ("batch", "sgd", 1, 1.6), ("batch", "sgd", 1, 1.8), ("batch", "sgd", 1, 2),
        ("batch", "polyak", 4, .176777), ("batch", "polyak", 4, .21022428499105428),
        ("batch", "polyak", 32, .707107), ("batch", "polyak", 32, 1.681793),
        ("batch", "polyak", 128, 2), ("batch", "polyak", 128, 2.378414),
        ("batch", "polyak", 256, 1.414214), ("batch", "polyak", 256, 2.828427),
        ("gap", "polyak", .125, 1.414214), ("gap", "polyak", .125, 2.828427),
        ("gap", "nesterov", 1/128, .707107),
    ]
    table = ["| 配置 | LR | 稳定 | 失败 | 未知 |", "|---|---:|---:|---:|---:|"]
    for axis, algorithm, x, lr in focus:
        row = next(c for c in cells if c["axis"] == axis and c["algorithm"] == algorithm
                   and c["x"] == x and math.isclose(c["lr"], lr, rel_tol=LR_TOL))
        if axis == "batch":
            label = f"{algorithm}, B={int(x)}" + ("" if algorithm == "sgd" else ", gap=1/64")
        else:
            label = f"{algorithm}, B=4, gap=1/{round(1/x)}"
        if row["K"] == 4000:
            label += ", K=4000"
        table.append(f"| {label} | {lr:.6g} | {row['n_stable']} | {row['n_unstable']} | {row['n_unknown']} |")
    states = report["selected_state_counts"]
    text = f"""# CIFAR-10 稳定性实验汇总（2026-09-20）

已合并完整 seed-0 扫描、边界五 seed 细化以及全部 200 组定向补实验。
共读取 {report['input_summaries']} 份 summary；旧扫描去重后 {report['old_unique_trials']} 个试验，
新补实验替换其中 {report['unique_old_trials_superseded']} 个同配置/seed 记录，
最终保留 {report['unique_selected_trials']} 个试验：{states['stable']} 稳定、
{states['unstable']} 失败、{states['unknown']} 个未知。
新补实验本身是 200/200 完成，其中 {report['revised_original_bounded']}/200 满足稳定判据。
本版按用户指定口径，将未被补跑结果替代的旧参数范数提前停止记为不稳定。

## 图中如何读数

- 上排固定 gap=1/64，横轴 B；SGD 无动量。下排固定 B=4，横轴 gap=1-rho。
- 颜色表示**稳定 seed 数 / 总 seed 数**。黄色为 1，深蓝色为 0。
- 未被补跑结果替代的旧参数范数提前停止统一作为失败，计入分母。本数据集不再有由这些记录造成的灰色斜线。
- 原先有部分未知 seed 的格子也重新计算比例。
- 白色表示没有测量。白圆点为多个可判定 seed，白方块表示包含新版补实验。
- 紫色虚线列是 Nesterov、gap=1/128、K=4000；其余列 K=2000。两种时长未混作同一配置。
- `seed_coverage.png` 给出可判定 seed 数，`merged_cells.csv` 同时给出稳定、失败、未知和总 seed 数。
- SGD 下排各列复用同一组无动量 B=4 结果，不是独立重复实验；全局计数只计算一次。

## 本图统计口径与保留规则

完成 K 步且未触发保留的安全停止条件，同时最后 10 次 probe loss 的均值/L0 <= 2，
且最后 10 次均值/前 10 次均值 <= 1.25，才记为稳定。每 50 步评估一次。
新版不因参数范数超过初始范数的 100 倍而停止；梯度范数 > 1e6、probe loss/L0 > 100、
NaN/Inf 仍视为失败。这里是有限 K 的经验 loss 判据，不等于无限时长稳定性，也不代表有效学习。

所有完成训练的旧结果都由逐次 probe loss 重新计算 10+10 窗口；共
{report['window_reclassification_count']} 个旧唯一试验因此改变标签，变化方向见 `window_reclassifications.csv`。
旧参数范数停止没有后续 loss；按本次指定的统计口径，未被新版替代的此类记录记为失败。
这是图表的分类约定，不声称观测到了这些旧轨迹在后续训练中的 loss 发散。
新版同配置、同 seed 的结果优先，无论它稳定还是失败；不选择有利 seed、不强制单调、不拟合预期直线。

LR 中有几处小数截断差异，例如 1.189207 与 1.189207299002155。
相对差 <= 1e-6 的值作为同一个绘图网格位置；`raw_lr` 和替换对照表保留原值。
这些近似相等 LR 的试验不声称是精确重现同一轨迹。

在 n 个 seed 中有 s 个稳定时，本版图色为 s/n，旧参数停止包含在 n 内。
表中仍保留上下界字段，本数据集中两者均等于该比例，**不是统计置信区间**。
补点由旧结果定向选择、各格 seed 数不同，颜色是描述性汇总。

## 重点位置的实际结果

""" + "\n".join(table) + """

SGD 的 B=1、LR=2 在新版 8 个 seed 中全部失败，不能把这一列强行补成与其他 B 一样高。
Polyak 的两个低 LR 小缺口在统一窗口后消失，但 B=32、LR=0.707107 的 seed-0 probe 失败仍存在。
Polyak 部分较大 B 的新版结果越过了旧参数停止边界；有新版结果的点优先采用新版，没有替代的旧参数停止在本图计为失败。
Nesterov 新增 gap=1/128 的六个 LR（0.25 至 0.707107）均为 5/5 稳定，
只表明这批 LR 在 K=4000 下通过；没有夹住失稳上界，不能据此确定临界 LR 或线性斜率。

## 文件

- `cifar_stability_merged.png`：六面板高清主图。
- `cifar_stability_merged.pdf`：可用于论文排版的 PDF（文字为矢量）。
- `seed_coverage.png`：可判定 seed 数的配套图。
- `merged_cells.csv`：每个绘图格子的计数、比例和未知结局范围。
- `selected_trials.csv`：去重后的试验逐条结论。
- `all_source_trials.csv`：所有输入记录及是否被选入。
- `replaced_trials.csv`：新结果替换旧结果的来源及原始 LR。
- `window_reclassifications.csv`：5+5 改为 10+10 后标签变化的旧试验。
- `parameter_stop_failures.csv`：本版明确计为失败的旧参数停止记录。
- `unresolved_parameter_stops.csv`：为兼容旧版保留的未知记录表，本次仅有表头。
- `summary.json`：计数、数据集与代码校验信息、汇总规则。
- `plot_cifar_summary.py`：唯一一个绘图/汇总代码文件。

## 本地复现

需要 Python 3、NumPy、Matplotlib，不需要 CUDA，不会重新训练。

```bash
python3 plot_cifar_summary.py \\
  --input /path/to/cifar_final_results_20260920_095122_403154Z.zip \\
  --output /path/to/new_summary
```

也可以把 `--input` 指向服务器项目目录 `/home/jnwang/momentum/cifar10_stability`，
脚本会读取其 `output/` 下的原始目录。输出请使用独立的新文件夹。
所有原始 summary、manifest、history 均只读。
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    write_csv(output / "parameter_stop_failures.csv", [r for r in selected if r["stop_reason"] == "parameter_safety_limit"])
    write_csv(output / "unresolved_parameter_stops.csv", [r for r in selected if r["state"] == "unknown"],
              fieldnames=list(selected[0]) if selected else None)


def plot(cells, output, coverage=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import NullLocator, FixedLocator, FuncFormatter
    import numpy as np

    columns = defaultdict(list)
    for row in cells:
        columns[(row["axis"], row["algorithm"], row["x"])].append(row)
    edges = {}
    for key, rows in columns.items():
        rows.sort(key=lambda r: r["lr"])
        if len({r["K"] for r in rows}) != 1:
            raise ValueError("Mixed horizons in one column")
        edges[key] = log_edges([r["lr"] for r in rows])
    ymin = min(v[0] for v in edges.values())
    ymax = max(v[-1] for v in edges.values())
    cmap = plt.get_cmap("viridis", 9) if coverage else plt.get_cmap("cividis")
    norm = Normalize(0, 8) if coverage else Normalize(0, 1)
    names = {"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}
    settings = {"font.size": 11, "font.family": "DejaVu Sans", "axes.titlesize": 14,
                "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                "axes.linewidth": 0.8, "hatch.linewidth": 0.35,
                "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(17.4, 12.4))
        grid = fig.add_gridspec(2, 3, left=0.065, right=0.91, bottom=0.19, top=0.86,
                               hspace=0.50, wspace=0.21)
        for i, axis in enumerate(("batch", "gap")):
            for j, algorithm in enumerate(ALGORITHMS):
                ax = fig.add_subplot(grid[i, j])
                ax.set_box_aspect(4/5)
                xs = sorted(key[2] for key in columns if key[:2] == (axis, algorithm))
                ax.set_yscale("log", base=2)
                ax.set_ylim(ymin, ymax)
                ax.set_facecolor("#ffffff")
                for ix, x in enumerate(xs):
                    key = (axis, algorithm, x)
                    rows = columns[key]
                    ys = edges[key]
                    vals = [r["n_assessed"] if coverage else r["stable_fraction_assessed"] for r in rows]
                    vals = np.array([np.nan if v is None else v for v in vals], float)
                    ax.pcolormesh([ix-0.5, ix+0.5], ys, np.ma.masked_invalid(vals[:, None]),
                                  cmap=cmap, norm=norm, shading="flat", rasterized=True)
                    for k, row in enumerate(rows):
                        if row["n_unknown"]:
                            face = "#e1e4e8" if not row["n_assessed"] else "none"
                            ax.add_patch(Rectangle((ix-0.5, ys[k]), 1, ys[k+1]-ys[k],
                                facecolor=face, hatch="///", edgecolor="#a2a8b0", linewidth=0, zorder=2))
                        if not coverage and row["n_revised"]:
                            ax.scatter(ix, row["lr"], s=15, marker="s", facecolors="white",
                                       edgecolors="#151b24", linewidths=0.5, zorder=3)
                        elif not coverage and row["n_assessed"] >= 2:
                            ax.scatter(ix, row["lr"], s=7, marker="o", facecolors="white",
                                       edgecolors="#151b24", linewidths=0.4, zorder=3)
                    if rows[0]["K"] == 4000:
                        ax.add_patch(Rectangle((ix-0.5, ymin), 1, ymax-ymin, fill=False,
                                               edgecolor="#8c3e8e", linewidth=1.4, linestyle="--", zorder=4))
                        ax.text(ix, 6.0, "$K=4000$", rotation=90, ha="center", va="top",
                                color="#8c3e8e", fontsize=9)
                ax.set_xlim(-0.5, len(xs)-0.5)
                ax.set_xticks(range(len(xs)))
                if axis == "batch":
                    labels = [str(int(x)) for x in xs]
                    subtitle = "No momentum" if algorithm == "sgd" else r"Fixed $1-\rho=1/64$"
                    ax.set_xlabel(r"Batch size $B$")
                else:
                    labels = [r"$1/" + str(round(1/x)) + r"$" for x in xs]
                    subtitle = r"Fixed $B=4$" + ("; no momentum" if algorithm == "sgd" else "")
                    ax.set_xlabel(r"Momentum gap $1-\rho$")
                ax.set_xticklabels(labels, fontsize=10)
                ax.yaxis.set_major_locator(FixedLocator([2**-8, 2**-6, 2**-4, 2**-2, 1, 4, 8]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3g}"))
                ax.yaxis.set_minor_locator(NullLocator())
                if j == 0:
                    ax.set_ylabel(r"Learning rate $\eta$")
                ax.set_title(f"({chr(97+3*i+j)})  {names[algorithm]}", loc="left", pad=23)
                ax.text(0.0, 1.035, subtitle, transform=ax.transAxes, fontsize=10, color="#515966")
                ax.tick_params(length=3, width=.7)
        cax = fig.add_axes([0.935, 0.26, 0.013, 0.53])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          ticks=list(range(9)) if coverage else [0, .2, .4, .6, .8, 1])
        cb.set_label("Number of seeds" if coverage else "Stable-run fraction", labelpad=10)
        cb.outline.set_linewidth(.6)
        title = "CIFAR-10 | Seed coverage" if coverage else "CIFAR-10 | Empirical risk stability"
        fig.suptitle(title, x=.49, y=.965, fontsize=20, fontweight="semibold")
        fig.text(.49, .920, "Full scan + boundary refinement + 200 targeted reruns", ha="center", fontsize=12, color="#3f4855")
        handles = [Patch(facecolor="white", edgecolor="#bdc2c9", label="No trial at this LR range")]
        if any(row["n_unknown"] for row in cells):
            handles.insert(0, Patch(facecolor="#e1e4e8", hatch="///", edgecolor="#a2a8b0", label="Incomplete result"))
        if not coverage:
            handles = [Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec="#151b24", label="Multiple assessed seeds"),
                       Line2D([], [], marker="s", ls="", ms=5, mfc="white", mec="#151b24", label="Includes revised reruns")] + handles
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.49, .095), ncol=len(handles),
                   frameon=False, fontsize=10, handlelength=1.5, columnspacing=1.8)
        fig.text(.49, .062, r"$K=2000$ except the purple column ($K=4000$). Tail loss / initial loss $\leq 2$; tail / previous window $\leq 1.25$.",
                 ha="center", fontsize=10, color="#414954")
        fig.text(.49, .035, "Both windows contain 10 evaluations. Old parameter-norm stops count as unstable; revised reruns take precedence.",
                 ha="center", fontsize=9.5, color="#414954")
        name = "seed_coverage" if coverage else "cifar_stability_merged"
        fig.savefig(output / (name + ".png"), dpi=220, facecolor="white")
        if not coverage:
            fig.savefig(output / (name + ".pdf"), facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if (output / "manifest.json").exists() or (output / "trials").exists():
        raise ValueError("The output must not be a training-result directory")
    if args.input.is_dir():
        for source in SOURCES:
            root = (args.input / source).resolve()
            if output == root or output in root.parents or root in output.parents:
                raise ValueError("Keep the output separate from original trial folders")
    bundle = Input(args.input)
    try:
        records, views, report = load_records(bundle)
    finally:
        bundle.close()
    selected, replacements, merge_report = merge_records(records)
    cells = build_cells(selected, views)
    window_changes, change_report = summarize_changes(records, selected, replacements)
    report.update(merge_report)
    report.update(change_report)
    report.update({"analysis": "10-evaluation windows; unreplaced old parameter-norm stops count as unstable",
                   "legacy_parameter_stop_policy": "count_as_unstable_as_requested",
                   "legacy_parameter_stops_counted_unstable": sum(r["stop_reason"] == "parameter_safety_limit" for r in selected),
                   "selection": "record-mode result takes precedence, independent of outcome; seed counts once",
                   "fraction_definition": "stable / total selected seeds; old parameter-only stops are counted in the unstable denominator",
                   "interval_definition": "[stable / total, (stable + unknown) / total]; descriptive bounds, not confidence intervals",
                   "caution": "Adaptive targeted selection and unequal seed counts: colors are descriptive, not unbiased stability probabilities",
                   "lr_note": "Differences <=1e-6 relative are grouped as rounded scan locations; preserve raw LR; not exact-replicate equivalence",
                   "horizon_note": "K=4000 gap=1/128 is separate and explicitly marked; no mixing with K=2000",
                   "rendering": "Per-column geometric LR bins centered on measured LR locations; no fitted/interpolated outcomes",
                   "plot_cells": len(cells), "plot_cells_with_unknown_seeds": sum(c["n_unknown"] > 0 for c in cells),
                   "plot_cells_without_assessed_seeds": sum(c["n_assessed"] == 0 for c in cells),
                   "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "merged_cells.csv", cells)
    write_csv(output / "selected_trials.csv", selected)
    write_csv(output / "all_source_trials.csv", records)
    write_csv(output / "replaced_trials.csv", replacements)
    write_csv(output / "window_reclassifications.csv", window_changes)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_notes(output, report, cells, selected)
    plot(cells, output)
    plot(cells, output, coverage=True)
    print(json.dumps({key: report[key] for key in (
        "input_summaries", "unique_selected_trials", "selected_state_counts", "revised_trials",
        "revised_original_bounded", "old_parameter_stops_rerun", "rerun_old_parameter_stop_outcomes",
        "window_reclassification", "plot_cells_with_unknown_seeds")}, indent=2))
    print("Saved:", output / "cifar_stability_merged.png")


if __name__ == "__main__":
    main()
