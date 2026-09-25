"""Regression coverage for repository-relative agent locations (issue #6)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from explorers.base import ExplorerResult
from explorers.opencode import OpenCodeExplorer
from explorers.parsing import iter_events, _resolve_repo_path, parse_relevant_files


class RelevantFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'Users/me/repos/demo'
        self.page = 'entry/src/main/ets/pages/Index.ets'
        for name in (self.page, 'Index.ets', 'src/B.ets'):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('// ArkTS source\n' * 30, encoding="utf-8")

    def parse(
        self, location: str, structured: bool = True, *, top_k: int | None = None,
    ) -> list[ExplorerResult]:
        text = (
            f'RELEVANT_FILES:\n- {location}'
            if structured else f'Look at {location} for the bug.'
        )
        return parse_relevant_files(text, 'case', repo_path=self.root, top_k=top_k)

    def assert_region(
        self, results: list[ExplorerResult], path: str, start: int, end: int,
    ) -> None:
        self.assertEqual(len(results), 1)
        region = results[0].regions[0]
        self.assertEqual((region.path, region.start, region.end), (path, start, end))

    def test_locations(self):
        cases = [
            ('Index.ets:10', 'Index.ets', 10, 10),
            (f'./{self.page}:3-7', self.page, 3, 7),
            (f'{self.root}/{self.page}:1-2', self.page, 1, 2),
            (self.page.replace('/', '\\') + ':3-7', self.page, 3, 7),
            (str(self.root / self.page).replace('/', '\\') + ':10', self.page, 10, 10),
            (f'demo/{self.page}:3-7', self.page, 3, 7),
        ]
        for structured in (True, False):
            for location, path, start, end in cases:
                with self.subTest(location=location, structured=structured):
                    results = self.parse(location, structured)
                    self.assert_region(results, path, start, end)

    def test_range_whitespace(self):
        for structured in (True, False):
            for location in ('Index.ets: 10-20', 'Index.ets:10 - 20',
                             'Index.ets : 10 - 20', 'Index.ets:10- 20'):
                with self.subTest(location=location, structured=structured):
                    results = self.parse(location, structured)
                    self.assert_region(results, 'Index.ets', 10, 20)

    def test_decorated_locations(self):
        for structured in (True, False):
            for location in ('**Index.ets:1-2**', 'Index.ets:1-2 (root cause)',
                             'Index.ets:1-2 # explanation', 'Index.ets:1-2!',
                             'Index.ets:1-2?', '"Index.ets":1-2',
                             '`Index.ets`:1-2', '"Index.ets:1-2"',
                             '**Index.ets:1-2** (root cause)',
                             'Index.ets:1-2 (root cause).', '"Index.ets:1-2."'):
                with self.subTest(location=location, structured=structured):
                    results = self.parse(location, structured)
                    self.assert_region(results, 'Index.ets', 1, 2)

    def test_numbered_list(self):
        text = 'RELEVANT_FILES:\n1. Index.ets:10-20\n2. src/B.ets:1-2'
        results = parse_relevant_files(text, 'case', repo_path=self.root)
        self.assertEqual([r.regions[0].path for r in results], ['Index.ets', 'src/B.ets'])

    def test_rootless_extensionless_ranges_and_empty_roots(self):
        for root in (None, '', '  '):
            for name in ('Makefile', 'Dockerfile', 'absent.ets'):
                with self.subTest(root=root, name=name):
                    results = parse_relevant_files(
                        f'RELEVANT_FILES:\n- {name}:10-20', 'case', repo_path=root,
                    )
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0].regions[0].path, name)
        # Direct resolver calls must also never accidentally inspect cwd.
        self.assertIsNone(_resolve_repo_path('README.md', ''))
        self.assertIsNone(_resolve_repo_path('README.md', '  '))

    def test_container_paths_resolve_to_existing_checkout_files(self):
        for prefix in ('/testbed/', '/workspace/demo/', '/opt/runner/repos/demo/'):
            with self.subTest(prefix=prefix):
                self.assert_region(self.parse(prefix + self.page + ':10'), self.page, 10, 10)
        with self.assertLogs('explorers.parsing', level='WARNING'):
            self.assertEqual(self.parse('/testbed/missing.ets:10'), [])

    def test_existing_outside_path_or_directory_is_not_remapped(self):
        outside = Path(self.tmp.name) / 'runner/repos/demo/Index.ets'
        outside.parent.mkdir(parents=True)
        outside.touch()
        with self.assertLogs('explorers.parsing', level='WARNING'):
            self.assertEqual(self.parse(str(outside) + ':10'), [])
        outside.unlink()
        outside.mkdir()
        with self.assertLogs('explorers.parsing', level='WARNING'):
            self.assertEqual(self.parse(str(outside) + ':10'), [])

    def test_prose_outside_rejected_block_is_recovered(self):
        for invalid in ('missing.ets:10', 'Index.ets:20-5', 'Index.ets:10-'):
            for before in (True, False):
                with self.subTest(invalid=invalid, before=before):
                    prose = 'The problem is in Index.ets:10.\n\n'
                    block = f'RELEVANT_FILES:\n- {invalid}\n\n'
                    text = prose + block if before else block + prose
                    with self.assertLogs('explorers.parsing', level='WARNING'):
                        results = parse_relevant_files(text, 'case', repo_path=self.root)
                    self.assert_region(results, 'Index.ets', 10, 10)

    def test_valid_block_takes_priority_over_prose(self):
        text = 'See Index.ets:10.\nRELEVANT_FILES:\n- src/B.ets:1-2'
        self.assert_region(
            parse_relevant_files(text, 'case', repo_path=self.root), 'src/B.ets', 1, 2,
        )

    def test_line_labels_columns_and_trailing_notes(self):
        for structured in (True, False):
            for suffix, start, end in (
                (':L10-L20', 10, 20), (':l10', 10, 10), (':10–20', 10, 20),
                (':10 — L20', 10, 20), (':10:3', 10, 10),
                (':10-20:3', 10, 20), (':10-20:', 10, 20),
                (':10-20 root cause', 10, 20),
            ):
                with self.subTest(suffix=suffix, structured=structured):
                    self.assert_region(self.parse('Index.ets' + suffix, structured), 'Index.ets', start, end)

    def test_apostrophes_in_prose_do_not_swallow_locations(self):
        text = "It's in Index.ets:10 and that's the cause."
        self.assert_region(parse_relevant_files(text, 'case', repo_path=self.root), 'Index.ets', 10, 10)
        self.assert_region(self.parse("'Index.ets':10", structured=False), 'Index.ets', 10, 10)

    def test_single_quoted_paths_with_spaces_stay_intact(self):
        local = self.root / 'Project/Index.ets'
        local.parent.mkdir()
        local.touch()
        outside = Path(self.tmp.name) / 'My Project/Index.ets'
        outside.parent.mkdir()
        outside.touch()
        with self.assertLogs('explorers.parsing', level='WARNING'):
            self.assertEqual(self.parse(f"'{outside}':10", structured=False), [])
        inside = self.root / 'My Project/Index.ets'
        inside.parent.mkdir()
        inside.touch()
        self.assert_region(
            self.parse("'My Project/Index.ets':10", structured=False),
            'My Project/Index.ets', 10, 10,
        )

    def test_whole_file(self):
        self.assert_region(self.parse(self.page), self.page, 1, -1)

    def test_invalid_ranges_are_logged_and_skipped(self):
        for structured in (True, False):
            for suffix in ('20-5', '0-5', '10-', '10 -', '20 - 5', '10 - nope', 'L20-L5', '10–', '10 —', '-1', '10--2'):
                with self.subTest(structured=structured, suffix=suffix):
                    with self.assertLogs('explorers.parsing', level='WARNING') as logs:
                        self.assertEqual(self.parse('src/B.ets:' + suffix, structured), [])
                    self.assertIn('Invalid line range', '\n'.join(logs.output))
                    self.assertIn('case', '\n'.join(logs.output))

    def test_unresolvable_paths_are_logged_and_skipped(self):
        outside = Path(self.tmp.name) / 'outside.ets'
        outside.touch()
        (self.root / 'escape.ets').symlink_to(outside)
        for path in (
            'missing.ets', str(outside), '../../../../outside.ets', 'escape.ets',
            'entry', 'C:\\other\\Index.ets',
        ):
            with self.subTest(path=path):
                with self.assertLogs('explorers.parsing', level='WARNING') as logs:
                    self.assertEqual(self.parse(path + ':1-2'), [])
                self.assertIn('Unresolvable path', '\n'.join(logs.output))

    def test_fallback_does_not_resolve_a_suffix_of_an_outside_path(self):
        with self.assertLogs('explorers.parsing', level='WARNING'):
            self.assertEqual(self.parse('/tmp/outside@src/B.ets:10', structured=False), [])

    def test_invalid_entries_do_not_consume_top_k(self):
        with self.assertLogs('explorers.parsing', level='WARNING'):
            results = self.parse('missing.ets:1-2\n- Index.ets:10\n- src/B.ets:2-4', top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].regions[0].path, 'Index.ets')

    def test_cli_passes_actual_root(self):
        output = f'RELEVANT_FILES:\n- {self.root}/{self.page}:10'
        with patch(
            'explorers._cli_agent_base.run_cli',
            return_value=subprocess.CompletedProcess([], 0, output, ''),
        ):
            results = OpenCodeExplorer(repo_root=self.root).explore(
                instance_id='case', query='find bug',
            )
        self.assert_region(results, self.page, 10, 10)

    def test_claude_and_cursor_pass_actual_root(self):
        from explorers.claude_code import ClaudeCodeExplorer
        from explorers.cursor_agent import CursorAgentExplorer

        output = f'RELEVANT_FILES:\n- {self.root}/{self.page}:10'
        for cls, module in ((ClaudeCodeExplorer, 'claude_code'),
                            (CursorAgentExplorer, 'cursor_agent')):
            with self.subTest(explorer=cls.__name__):
                with patch(f'explorers.{module}.run_cli',
                           return_value=subprocess.CompletedProcess([], 0, output, '')):
                    results = cls(repo_root=self.root).explore(
                        instance_id='case', query='find bug',
                    )
                self.assert_region(results, self.page, 10, 10)

    def test_awe_passes_actual_root_for_text_and_finish_payload(self):
        from explorers.awe_agent_explorer import AweAgentExplorer

        absolute = str(self.root / self.page)
        for output, finish in ((f'RELEVANT_FILES:\n- {absolute}:10', {}),
                               ('', {absolute: [10]})):
            with self.subTest(finish=bool(finish)):
                with patch.object(AweAgentExplorer, '_run_agent',
                                  new=AsyncMock(return_value=(output, finish))):
                    results = AweAgentExplorer(repo_root=self.root).explore(
                        instance_id='case', query='find bug',
                    )
                self.assert_region(results, self.page, 10, 10)

    def test_mini_swe_passes_actual_root(self):
        from explorers.mini_swe_agent_explorer import MiniSWEAgentExplorer

        output = f'RELEVANT_FILES:\n- {self.root}/{self.page}:10'

        class FakeModel:
            def __init__(self, **kwargs):
                self.config = SimpleNamespace()

        for submission in (output, ''):
            with self.subTest(submission=bool(submission)):
                agent = SimpleNamespace(run=lambda **kw: {'submission': submission},
                                        messages=[{'content': output}])
                modules = {}
                for name, attrs in {
                    'agents.default': {'DefaultAgent': lambda *args, **kw: agent},
                    'environments.local': {'LocalEnvironment': lambda **kw: None},
                    'models.litellm_model': {'LitellmModel': FakeModel},
                    'models.litellm_textbased_model': {'LitellmTextbasedModel': FakeModel},
                }.items():
                    module = ModuleType('minisweagent.' + name)
                    module.__dict__.update(attrs)
                    modules[module.__name__] = module
                with patch.dict(sys.modules, modules):
                    results = MiniSWEAgentExplorer(repo_root=self.root).explore(
                        instance_id='case', query='find bug',
                    )
                self.assert_region(results, self.page, 10, 10)

    def test_legacy_rootless_calls_still_work(self):
        results = parse_relevant_files('RELEVANT_FILES:\n- src/main.py:10-20', 'case')
        self.assertEqual(results[0].regions[0].path, 'src/main.py')


if __name__ == '__main__':
    unittest.main()


class IterEventsTest(unittest.TestCase):
    """One JSON-lines reader serves the answer extractor, the usage scanner
    and the tests, so all three see the same events."""

    def test_yields_only_json_objects(self):
        raw = "\n".join([
            "banner line",
            '{"type": "text", "part": {"text": "hi"}}',
            "[1, 2]",
            "{not json",
            "   ",
            ' {"type": "step_finish"} ',
        ])
        self.assertEqual(
            list(iter_events(raw)),
            [{"type": "text", "part": {"text": "hi"}}, {"type": "step_finish"}],
        )

    def test_bom_and_array_events_are_retained(self):
        raw = '\ufeff {"type": "step_start"}\n[{"type": "text"}, 1, null]'
        self.assertEqual(list(iter_events(raw)),
                         [{"type": "step_start"}, {"type": "text"}])

    def test_empty_and_none_input_yield_nothing(self):
        self.assertEqual(list(iter_events("")), [])
        self.assertEqual(list(iter_events(None)), [])
