#!/usr/bin/env python3
"""Real subprocess end-to-end tests for the public ``run.py`` CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run.py"
EXPECTED_NVIDIA_CAPTURE_STDERR = (
    "run.py capture is not available on NVIDIA cards (h20/b300/sm86) any more.\n"
    "Use the main audit, which runs nsys with GPU metrics itself:\n"
    '  MCP tool weaveprof_megakernel_audit with "capture_nsys": true, or\n'
    "  python3 -B weaveprof/aka_adapter.py megakernel-audit < request.json\n"
    "To capture an arbitrary program with nsys only, run nsys directly (see SKILL.md).\n"
)


class RunCliTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-run-cli-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.env = os.environ.copy()
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.cache_before = set(ROOT.rglob("__pycache__"))
        self.addCleanup(self._assert_no_bytecode_was_written)

    def _assert_no_bytecode_was_written(self) -> None:
        self.assertEqual(self.cache_before, set(ROOT.rglob("__pycache__")))
        self.assertEqual([], list(self.root.rglob("__pycache__")))

    def run_cli(self, argv: Sequence[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(RUN), *argv],
            cwd=str(cwd or self.project),
            env=self.env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )

    def run_script(self, script: Path, argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(script), *argv],
            cwd=str(cwd),
            env=self.env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )

    @staticmethod
    def command_diagnostic(completed: subprocess.CompletedProcess[str]) -> str:
        return "exit={}\nstdout:\n{}\nstderr:\n{}".format(
            completed.returncode, completed.stdout, completed.stderr
        )

    def assert_success(self, completed: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(0, completed.returncode, self.command_diagnostic(completed))

    def test_gemm_probe_dispatches_to_the_packaged_probe(self) -> None:
        completed = self.run_cli(("gemm-probe", "--help"))
        self.assert_success(completed)
        self.assertTrue(completed.stdout.lstrip().startswith("usage: gemm_probe.py"))

    def test_capture_routes_amd_help_without_starting_gpu_work(self) -> None:
        mi308x = self.run_cli(
            (
                "capture",
                "--card",
                "mi308x",
                "--arch",
                "gfx942",
                "--project",
                str(self.project),
                "--",
                "--help",
            )
        )
        self.assert_success(mi308x)
        self.assertTrue(mi308x.stdout.lstrip().startswith("usage: capture_rocm.py"))

    def test_capture_rejects_nvidia_cards_and_guides_to_megakernel_audit(self) -> None:
        cases = (
            ("h20_bare", ("capture", "--card", "h20")),
            ("h20_full", ("capture", "--card", "h20", "--arch", "sm_90", "--project", str(self.project), "--collector", "nsys", "--", "--help")),
            ("b300_bare", ("capture", "--card", "b300")),
            ("b300_sm100a", ("capture", "--card", "b300", "--arch", "sm_100a", "--project", str(self.project), "--", "--help")),
            ("b300_sm103a", ("capture", "--card", "b300", "--arch", "sm_103a", "--project", str(self.project), "--", "--help")),
            ("sm86_bare", ("capture", "--card", "sm86")),
            ("sm86_full", ("capture", "--card", "sm86", "--arch", "sm_86", "--project", str(self.project), "--", "--help")),
        )
        for label, argv in cases:
            with self.subTest(case=label):
                completed = self.run_cli(argv)
                self.assertEqual(2, completed.returncode, self.command_diagnostic(completed))
                self.assertEqual("", completed.stdout, self.command_diagnostic(completed))
                self.assertEqual(EXPECTED_NVIDIA_CAPTURE_STDERR, completed.stderr, self.command_diagnostic(completed))


if __name__ == "__main__":
    unittest.main()
