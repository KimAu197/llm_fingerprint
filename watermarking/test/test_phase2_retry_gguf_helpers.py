import importlib.util
import unittest
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "run_pairwise_overlap_phase2_retry_gguf.py"
SPEC = importlib.util.spec_from_file_location("phase2_retry_gguf", SCRIPT)
phase2_retry_gguf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase2_retry_gguf)


@dataclass(frozen=True)
class Row:
    name: str
    tokenizer_from: str
    gguf_path: str


class Phase2RetryGgufHelpersTest(unittest.TestCase):
    def test_filter_rows_by_filename_regex(self):
        rows = [
            Row("bf16", "tok", "Model-BF16.gguf"),
            Row("q4", "tok", "Model-Q4_K_M.gguf"),
            Row("f16", "tok", "Model-F16.gguf"),
        ]
        args = Namespace(
            include_gguf_filename_regex=None,
            exclude_gguf_filename_regex="BF16|F16",
            max_targets=None,
        )

        filtered = phase2_retry_gguf._filter_gguf_rows_for_cli(rows, args)

        self.assertEqual([r.name for r in filtered], ["q4"])

    def test_target_only_prompt_scope_uses_only_target_fingerprints(self):
        models = ["old/a", "new/q4", "new/q5"]
        targets = ["new/q4", "new/q5"]
        all_fps = {
            "old/a": ["old-fp"],
            "new/q4": ["q4-fp"],
            "new/q5": ["q5-fp"],
        }
        args = Namespace(target_prompt_scope="target_only")

        prompts = phase2_retry_gguf._prompts_for_scope(models, targets, all_fps, args)

        self.assertEqual(prompts, ["q4-fp", "q5-fp"])


if __name__ == "__main__":
    unittest.main()
