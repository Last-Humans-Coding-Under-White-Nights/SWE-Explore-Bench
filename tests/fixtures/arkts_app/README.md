# ArkTS localization fixture

This small project was written by hand for tests in this repository. It is
synthetic test scaffolding, not scraped code or a benchmark dataset. It does
not need a HarmonyOS SDK and is not a complete buildable application.

`Index.ets` contains an ArkUI page with `@Entry`, `@Component`, `@State`, and
a `build()` method. It imports `DataSource.ets`, which imports the shared
`CatalogItem` declaration from `types.d.ets`. The two configuration files
exercise JSON5 comments, unquoted keys, single quotes, and trailing commas.

The code deliberately contains two independent bugs: the Next item handler
assigns the selected index to itself, and `getTitle` adds one to the requested
index. Preserve these bugs so that localization tests have known answers.

`expected_locations.json` is an array of example cases. Each case has a unique
`instance_id`, a bug-report `query`, and expected `regions`. Region `path`
values are relative to this directory with forward slashes. `start` and `end`
are **one-based, inclusive** line numbers, matching `ContextRegion`. `snippet`
is the exact selected source text, joined with newlines and without a final
newline. When editing the source, update the expected ranges and snippets
together.

From the repository root, run:

```sh
python3 -m unittest discover -s tests -p 'test_arkts_fixture.py'
```

The tests check the answers against the files and exercise structured and
free-text explorer-output parsing, including file-only `.ets` and `.json5`
locations. Bare `pytest` runs the unit tests in `tests/`; optional quality tests
can be selected explicitly with `pytest quality/tests/`.

These tests do not run ArkTS code or measure an explorer's ability to solve the
example queries. Other tests can use this directory as their repository root
and load the same JSON answers.
