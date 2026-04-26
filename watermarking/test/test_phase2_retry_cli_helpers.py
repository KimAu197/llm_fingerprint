import csv
import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "run_pairwise_overlap_phase2_retry.py"
SPEC = importlib.util.spec_from_file_location("phase2_retry", SCRIPT)
phase2_retry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase2_retry)


class Phase2RetryCliHelpersTest(unittest.TestCase):
    def test_phase2_targets_can_be_loaded_from_csv_and_cli(self):
        with tempfile.TemporaryDirectory() as td:
            target_csv = Path(td) / "targets.csv"
            with target_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model_id"])
                writer.writerow(["org/model-b"])
                writer.writerow(["org/model-c"])

            args = Namespace(
                phase2_models="org/model-a,org/model-b",
                phase2_models_csv=str(target_csv),
            )

            targets = phase2_retry._phase2_targets_from_cli(args)

        self.assertEqual(targets, ["org/model-a", "org/model-b", "org/model-c"])

    def test_existing_labeled_matrix_models_are_prepended_for_csv_targets(self):
        with tempfile.TemporaryDirectory() as td:
            matrix = Path(td) / "overlap_matrix.csv"
            with matrix.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["", "old/a", "old/b"])
                writer.writerow(["old/a", "1.0", "0.1"])
                writer.writerow(["old/b", "0.2", "1.0"])

            models = phase2_retry._models_with_existing_labeled_matrix_seed(
                csv_models=["new/c"],
                target_models=["new/d"],
                existing_overlap_matrix=str(matrix),
                use_phase2_models_csv=True,
            )

        self.assertEqual(models, ["old/a", "old/b", "new/c", "new/d"])

    def test_gpu_ids_are_parsed_and_deduped(self):
        args = Namespace(gpu_ids="0,1,2,1")

        gpu_ids = phase2_retry._gpu_ids_from_cli(args)

        self.assertEqual(gpu_ids, [0, 1, 2])

    def test_models_are_partitioned_round_robin_by_gpu(self):
        partitions = phase2_retry._partition_models_by_gpu(
            ["m0", "m1", "m2", "m3", "m4"],
            [0, 1],
        )

        self.assertEqual(partitions, [(0, ["m0", "m2", "m4"]), (1, ["m1", "m3"])])


if __name__ == "__main__":
    unittest.main()
