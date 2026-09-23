#!/usr/bin/env python3
"""Fail-closed launcher/preflight for the EOS Q3/R5 audited training command."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r4_training_launcher as base  # noqa: E402


LAUNCH_AUDIT_SCHEMA = "eos.q3_r5_broadfocus.training_launch_audit.v1"
PREFLIGHT_SCHEMA = "eos.q3_r5_training_launcher_preflight.v1"
EXPECTED_PARSED_WORKLOAD = {
    "train": 294,
    "batch": 4,
    "steps_per_epoch": 74,
    "train_pairs_per_epoch": 127469,
    "planned_pairs": 254938,
    "actual_train_pairs": 0,
}
EXPECTED_LEGAL_SCOPE = dict(base.EXPECTED_LEGAL_SCOPE)
EXPECTED_ARGV_DELTA = dict(base.EXPECTED_ARGV_DELTA)


def _patch_base_constants() -> None:
    base.LAUNCH_AUDIT_SCHEMA = LAUNCH_AUDIT_SCHEMA
    base.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    base.EXPECTED_PARSED_WORKLOAD = dict(EXPECTED_PARSED_WORKLOAD)
    base.EXPECTED_LEGAL_SCOPE = dict(EXPECTED_LEGAL_SCOPE)
    base.EXPECTED_ARGV_DELTA = dict(EXPECTED_ARGV_DELTA)


def parse_args(argv: list[str] | None = None):
    return base.parse_args(argv)


def preflight(args):
    _patch_base_constants()
    return base.preflight(args)


def main(argv: list[str] | None = None) -> int:
    _patch_base_constants()
    return base.main(argv)


PreflightError = base.PreflightError
audit_sha256 = base.audit_sha256
display_path = base.display_path
package_state = base.package_state
sha256_file = base.sha256_file
sha256_bytes = base.sha256_bytes
sibling_content_rollup = base.sibling_content_rollup
sibling_json_array_rollup = base.sibling_json_array_rollup
sibling_no_final_newline_rollup = base.sibling_no_final_newline_rollup


if __name__ == "__main__":
    raise SystemExit(main())
