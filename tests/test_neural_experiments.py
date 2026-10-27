"""CPU-only regression checks for the published neural-network experiments.

Run from a checkout: python -m unittest discover -s tests -v
Requires NumPy, pandas and Matplotlib; does not require PyTorch or datasets.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGENET = ROOT / "data_scaling/imagenet"
CIFAR = ROOT / "stability/cifar10"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class NeuralExperimentTests(unittest.TestCase):
    def assert_table_matches(self, expected, actual, keys):
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            for key in keys:
                x, y = left[key], right[key]
                if x == "" and y is None:
                    continue
                if str(x) == str(y):
                    continue
                try:
                    matches = math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-12)
                except (TypeError, ValueError):
                    matches = False
                self.assertTrue(matches, f"{key}: {x!r} != {y!r}")

    def run_entrypoint(self, module: Path, script: str, *args: str):
        result = subprocess.run([sys.executable, str(module / script), *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_original_sources_and_results(self):
        count = 0
        for root in (IMAGENET, CIFAR):
            for row in read_csv(root / "docs/import_manifest.csv"):
                payload = (root / row["destination"]).read_bytes()
                self.assertEqual(len(payload), int(row["bytes"]))
                self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])
                count += 1
        self.assertEqual(count, 826)

    def test_imagenet_grid_matches_recorded_runs(self):
        sweep = load_module("imagenet_sweep", IMAGENET / "sweep.py")
        trainer = load_module("imagenet_trainer", IMAGENET / "source/imagenet_momentum_experiment.py")
        rows = sweep.load_grid(IMAGENET / "configs/paper_grid.csv")
        self.assertEqual(len(rows), 396)
        aliases = {"micro_batch": "micro_batch_per_device", "sample_budget": "sample_budget_requested"}
        for row in rows:
            command = sweep.train_command(row, Path("data"), Path("outputs") / row["run_name"], 0)
            parsed = trainer.make_parser().parse_args(command[2:])
            original = json.loads((IMAGENET / "results/paper_3epochs" / row["run_name"] / "summary.json").read_text())
            for field in sweep.FIELDS:
                actual, expected = getattr(parsed, field), original[aliases.get(field, field)]
                if isinstance(actual, (float, int)):
                    self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
                else:
                    self.assertEqual(actual, expected)

    def test_imagenet_best_configuration_selection(self):
        module = load_module("imagenet_selection", IMAGENET / "source/imagenet_momentum_experiment.py")
        snapshot = IMAGENET / "results/paper_3epochs"
        summaries = module.load_summaries(snapshot)
        self.assertEqual(len(summaries), 396)
        aggregated = module.aggregate_configurations(summaries)
        selected = module.select_best_by_batch(aggregated, required_runs=1)
        self.assertEqual(len(selected), 39)
        for name, actual in (("aggregated_configurations.csv", aggregated), ("best_by_batch.csv", selected)):
            expected = read_csv(snapshot / name)
            self.assert_table_matches(expected, actual, expected[0].keys())

    def test_cifar_training_hashes(self):
        module = load_module("cifar_hash_audit", CIFAR / "source/plot_cifar_summary.py")
        hashes = module.audit_source_code(module.Input(CIFAR / "source"))
        self.assertEqual(len(hashes), 2)
        self.assertNotEqual(hashes["cifar_stability.py"], hashes["cifar_stability_revised.py"])

    def test_cifar_boundaries_and_fits(self):
        module = load_module("cifar_analysis", CIFAR / "source/plot_cifar_all.py")
        snapshot = CIFAR / "results/paper_2000steps"
        cells, excluded = module.load_cells(snapshot / "merged_cells.csv")
        self.assertEqual((len(cells), len(excluded)), (1309, 0))
        points, audit = module.boundaries(cells, 0.5)
        fits = module.fit_models(points)
        for name, actual in (("empirical_boundaries.csv", points), ("boundary_extraction.csv", audit), ("fit_parameters.csv", fits)):
            expected = read_csv(snapshot / name)
            self.assert_table_matches(expected, actual, expected[0].keys())

    def test_cli_help_and_default_dry_runs(self):
        for module in (IMAGENET, CIFAR):
            for script in ("train.py", "sweep.py", "plot.py"):
                self.run_entrypoint(module, script, "--help")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.run_entrypoint(IMAGENET, "sweep.py", "--runs-root", root / "imagenet")
            self.assertEqual(len(read_csv(root / "imagenet/reproduction_manifest.csv")), 396)
            self.assertEqual(list((root / "imagenet").glob("*/summary.json")), [])
            self.run_entrypoint(CIFAR, "sweep.py", "--output", root / "cifar")
            manifest = json.loads((root / "cifar/manifest.json").read_text())
            self.assertEqual(len(manifest["jobs"]), 4945)
            self.assertFalse((root / "cifar/trials").exists())

    def test_plot_entrypoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.run_entrypoint(IMAGENET, "plot.py", "--reaggregate", "--output", root / "imagenet", "--dpi", "60")
            self.assertTrue((root / "imagenet/fixed_D_optimal_loss_vs_batch.png").is_file())
            self.run_entrypoint(CIFAR, "plot.py", "--output", root / "cifar", "--dpi", "60")
            report = json.loads((root / "cifar/redraw_summary.json").read_text())
            self.assertFalse(report["raw_trials_reaudited"])
            self.assertTrue((root / "cifar/cifar_stability_merged.png").is_file())

    def test_archived_outputs_are_protected(self):
        for module, snapshot in ((IMAGENET, "paper_3epochs"), (CIFAR, "paper_2000steps")):
            result = subprocess.run([sys.executable, str(module / "plot.py"), "--output", str(module / "results" / snapshot)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("output directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
