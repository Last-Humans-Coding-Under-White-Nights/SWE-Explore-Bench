"""Keep the handwritten ArkTS localization answers usable by other tests."""
import json
import unittest
from pathlib import Path

from explorers.parsing import parse_file_paths, parse_relevant_files


FIXTURE = Path(__file__).parent / 'fixtures' / 'arkts_app'


class ArkTSFixtureTest(unittest.TestCase):
    def test_expected_locations_match_source(self):
        manifest = FIXTURE / 'expected_locations.json'
        self.assertTrue(manifest.is_file(), 'ArkTS localization answers are missing')
        cases = json.loads(manifest.read_text(encoding='utf-8'))
        self.assertTrue(cases)
        self.assertEqual(len({case['instance_id'] for case in cases}), len(cases))
        for case in cases:
            with self.subTest(case=case['instance_id']):
                self.assertTrue(case['query'].strip())
                self.assertTrue(case['regions'])
                for region in case['regions']:
                    path = Path(region['path'])
                    self.assertFalse(path.is_absolute())
                    self.assertNotIn('..', path.parts)
                    target_file = FIXTURE / path
                    self.assertTrue(target_file.is_file(), f'Fixture file not found: {path}')
                    lines = target_file.read_text(encoding='utf-8').splitlines()
                    start, end = region['start'], region['end']
                    self.assertGreaterEqual(start, 1)
                    self.assertGreaterEqual(end, start)
                    self.assertLessEqual(end, len(lines))
                    self.assertEqual('\n'.join(lines[start - 1:end]), region['snippet'])

    def test_agent_output_preserves_arkts_locations(self):
        cases = json.loads((FIXTURE / 'expected_locations.json').read_text(encoding='utf-8'))
        for case in cases:
            for structured in (True, False):
                with self.subTest(case=case['instance_id'], structured=structured):
                    locations = [
                        f"{r['path']}:{r['start']}-{r['end']}" for r in case['regions']
                    ]
                    text = ('RELEVANT_FILES:\n' + '\n'.join(f'- {p}' for p in locations)
                            if structured else 'Look at ' + ', '.join(locations))
                    results = parse_relevant_files(text, case['instance_id'])
                    for result in results:
                        self.assertEqual(result.instance_id, case['instance_id'])
                    actual = [(r.path, r.start, r.end)
                              for result in results for r in result.regions]
                    expected = [(r['path'], r['start'], r['end']) for r in case['regions']]
                    self.assertEqual(actual, expected)

    def test_file_only_arkts_locations(self):
        paths = [
            'build-profile.json5',
            'oh-package.json5',
            'entry/src/main/ets/pages/Index.ets',
            'entry/src/main/ets/common/types.d.ets',
        ]
        for path in paths:
            self.assertTrue((FIXTURE / path).is_file())
        for structured in (True, False):
            with self.subTest(structured=structured):
                text = ('RELEVANT_FILES:\n' + '\n'.join(f'- {p}' for p in paths)
                        if structured else 'Look at ' + ', '.join(paths))
                results = parse_file_paths(text, 'arkts-files')
                self.assertEqual(
                    [(r.path, r.start, r.end) for result in results for r in result.regions],
                    [(path, 1, -1) for path in paths],
                )

    def test_file_extensions_are_not_truncated(self):
        for extension in ('json5', 'json', 'jsx', 'js', 'tsx', 'ts', 'cpp', 'c'):
            with self.subTest(extension=extension):
                path = f'src/example.{extension}'
                results = parse_file_paths(f'Look at {path}.', 'file-extension')
                self.assertEqual(
                    [r.path for result in results for r in result.regions], [path],
                )
        self.assertEqual(parse_file_paths('Look at file.json5backup', 'unknown-extension'), [])

    def test_unstructured_json5_location(self):
        results = parse_relevant_files(
            'Look at build-profile.json5:4-6 for the product configuration.',
            'arkts-build-config',
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].instance_id, 'arkts-build-config')
        self.assertEqual(
            [(r.path, r.start, r.end) for r in results[0].regions],
            [('build-profile.json5', 4, 6)],
        )


if __name__ == '__main__':
    unittest.main()
