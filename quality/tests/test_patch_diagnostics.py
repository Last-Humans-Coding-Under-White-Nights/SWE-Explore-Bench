from __future__ import annotations

import unittest

from quality.gen_patches import _extract_diff
from quality.patch_diagnostics import (
    PATCH_FAILURE_EXTRACTION_FAILED,
    PATCH_FAILURE_FILE_OUTSIDE,
    PATCH_FAILURE_NO_DIFF_LIMITS_EXCEEDED,
    PATCH_FAILURE_REGION_OUTSIDE,
    PATCH_FAILURE_WITHIN_ALLOWED_BUT_LOST,
    collect_patch_candidates,
    diagnose_patch_flow,
    materialize_allowed_regions_for_contract,
    resolve_regions_with_content,
    select_best_patch_candidate,
)


class PatchDiagnosticsTest(unittest.TestCase):
    def test_extract_diff_trims_non_diff_preamble(self) -> None:
        raw = (
            "Wrote patch.txt with staged diff.\n"
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        extracted = _extract_diff(raw)
        self.assertTrue(extracted.startswith("diff --git a/foo.py b/foo.py\n"))
        self.assertNotIn("Wrote patch.txt with staged diff.", extracted)

    def test_collect_patch_candidates_covers_submission_and_tool_sources(self) -> None:
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        messages = [{"role": "tool", "content": f"<output>\n{patch}\n</output>", "extra": {"raw_output": patch}}]
        candidates = collect_patch_candidates(messages, {"submission": patch, "exit_status": "Submitted"})
        self.assertGreaterEqual(
            {candidate["source"] for candidate in candidates},
            {"info.submission", "tool.content", "tool.extra.raw_output"},
        )
        selected = select_best_patch_candidate(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["source"], "info.submission")

    def test_diagnose_no_diff_limits_exceeded(self) -> None:
        diagnosis = diagnose_patch_flow(
            instance_id="case-no-diff",
            messages=[],
            trajectory_info={"exit_status": "LimitsExceeded", "submission": ""},
            allowed_regions=[],
            final_prediction="",
        )
        self.assertFalse(diagnosis["patch_diagnosis"]["has_diff_candidate_raw"])
        self.assertEqual(diagnosis["patch_diagnosis"]["failure_mode"], PATCH_FAILURE_NO_DIFF_LIMITS_EXCEEDED)

    def test_diagnose_extraction_failure_when_diff_markers_exist_but_no_valid_patch(self) -> None:
        bad_output = (
            "diff --git a/foo.py b/foo.py\n"
            "this is not a valid unified diff body\n"
            "still missing headers and hunks\n"
        )
        diagnosis = diagnose_patch_flow(
            instance_id="case-extraction-fail",
            messages=[{"role": "tool", "content": "", "extra": {"raw_output": bad_output}}],
            trajectory_info={"exit_status": "Submitted", "submission": ""},
            allowed_regions=[],
            final_prediction="",
        )
        self.assertTrue(diagnosis["patch_diagnosis"]["has_diff_candidate_raw"])
        self.assertIsNone(diagnosis["patch_diagnosis"]["selected_diff_source"])
        self.assertEqual(diagnosis["patch_diagnosis"]["failure_mode"], PATCH_FAILURE_EXTRACTION_FAILED)

    def test_diagnose_file_and_region_outside_are_distinct(self) -> None:
        file_outside_patch = (
            "diff --git a/bar.py b/bar.py\n"
            "--- a/bar.py\n"
            "+++ b/bar.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        file_outside = diagnose_patch_flow(
            instance_id="case-file-outside",
            messages=[],
            trajectory_info={"exit_status": "Submitted", "submission": file_outside_patch},
            allowed_regions=[{"path": "foo.py", "start": 1, "end": 2, "content": "a\nb\n"}],
            final_prediction="",
        )
        self.assertEqual(file_outside["patch_diagnosis"]["file_check_result"], "file_outside_allowed_files")
        self.assertEqual(file_outside["patch_diagnosis"]["failure_mode"], PATCH_FAILURE_FILE_OUTSIDE)

        region_outside_patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -4 +4 @@\n"
            "-old\n"
            "+new\n"
        )
        region_outside = diagnose_patch_flow(
            instance_id="case-region-outside",
            messages=[],
            trajectory_info={"exit_status": "Submitted", "submission": region_outside_patch},
            allowed_regions=[{"path": "foo.py", "start": 1, "end": 2, "content": "a\nb\n"}],
            final_prediction="",
        )
        self.assertEqual(region_outside["patch_diagnosis"]["file_check_result"], "pass")
        self.assertEqual(region_outside["patch_diagnosis"]["region_check_result"], "file_allowed_but_region_outside")
        self.assertEqual(region_outside["patch_diagnosis"]["failure_mode"], PATCH_FAILURE_REGION_OUTSIDE)

    def test_end_minus_one_whole_file_region_resolves_to_eof(self) -> None:
        resolved = resolve_regions_with_content([
            {"path": "foo.py", "start": 1, "end": -1, "content": "line1\nline2\nline3\n"}
        ])
        self.assertEqual(resolved[0]["end"], 3)
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -3 +3 @@\n"
            "-line3\n"
            "+changed\n"
        )
        diagnosis = diagnose_patch_flow(
            instance_id="case-eof",
            messages=[],
            trajectory_info={"exit_status": "Submitted", "submission": patch},
            allowed_regions=[{"path": "foo.py", "start": 1, "end": -1, "content": "line1\nline2\nline3\n"}],
            final_prediction="",
        )
        self.assertEqual(diagnosis["patch_diagnosis"]["file_check_result"], "pass")
        self.assertEqual(diagnosis["patch_diagnosis"]["region_check_result"], "fully_within_allowed_regions")
        self.assertEqual(diagnosis["patch_diagnosis"]["failure_mode"], PATCH_FAILURE_WITHIN_ALLOWED_BUT_LOST)

    def test_materialize_contract_regions_merge_and_padding(self) -> None:
        regions = [
            {"path": "foo.py", "start": 10, "end": 12, "content": "a\nb\nc\n"},
            {"path": "foo.py", "start": 14, "end": 15, "content": "d\ne\n"},
        ]
        merged = materialize_allowed_regions_for_contract(regions, contract_mode="merged_spans_per_file")
        self.assertEqual(merged, [{"path": "foo.py", "start": 10, "end": 12}, {"path": "foo.py", "start": 14, "end": 15}])
        padded = materialize_allowed_regions_for_contract(regions, contract_mode="span_padding", span_padding=2)
        self.assertEqual(padded, [{"path": "foo.py", "start": 8, "end": 17}])

    def test_file_level_allow_contract_does_not_flag_region_outside(self) -> None:
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -20 +20 @@\n"
            "-old\n"
            "+new\n"
        )
        diagnosis = diagnose_patch_flow(
            instance_id="case-file-level-allow",
            messages=[],
            trajectory_info={"exit_status": "Submitted", "submission": patch},
            allowed_regions=[{"path": "foo.py", "start": 1, "end": 2, "content": "a\nb\n"}],
            final_prediction="",
            contract_mode="file_level_allow",
        )
        self.assertEqual(diagnosis["patch_diagnosis"]["file_check_result"], "pass")
        self.assertEqual(diagnosis["patch_diagnosis"]["region_check_result"], "file_level_allow_allows_any_line")
        self.assertEqual(diagnosis["patch_diagnosis"]["failure_mode"], PATCH_FAILURE_WITHIN_ALLOWED_BUT_LOST)


if __name__ == "__main__":
    unittest.main()
