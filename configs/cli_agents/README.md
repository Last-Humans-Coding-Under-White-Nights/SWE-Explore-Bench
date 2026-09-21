# OpenCode and DevEco Code evaluation profiles

Committed, credential-free configurations for running the `opencode` and
`deveco` explorers against ArkTS / OpenHarmony projects. Point the runner's
config-dir flag at one of these directories and every reproduction uses the
same model wiring, the same read-only permissions, and the same MCP setup.

```text
configs/cli_agents/
|-- opencode/arkts/opencode.json                 # semantic navigation over MCP: on
|-- opencode/arkts/checkout/.serena/project.yml  # placed into the checkout per run
|-- opencode/arkts-no-mcp/opencode.json          # semantic navigation over MCP: off
|-- deveco/arkts/deveco.json                     # same profile, DevEco Code filename
|-- deveco/arkts/checkout/.serena/project.yml
`-- deveco/arkts-no-mcp/deveco.json
```

DevEco Code is a HarmonyOS-oriented fork of OpenCode, so the two CLIs share
one configuration schema. The four JSON files differ only in filename and in
the `mcp.serena.enabled` flag; `tests/test_cli_agent_profiles.py` enforces
that. `.gitignore` admits exactly these four JSON paths, so a private profile
placed anywhere else, including beside them, stays ignored.

A profile directory may hold a `checkout/` subdirectory. The explorer copies
its files into the repository before each run, skipping any the checkout
already has, and removes the ones the run left unchanged afterwards. That is
how the MCP-on variants deliver Serena's project file.

## Supported CLI versions

| CLI | Version | Status |
| --- | --- | --- |
| OpenCode | 1.18.29 | Verified: honours `OPENCODE_CONFIG_DIR`, loads this profile, `run --auto --format json --dir` as the explorer sends it. |
| DevEco Code | 0.1.9 | The version the explorer's argv was written against (`--dangerously-skip-permissions`). |
| DevEco Code | 0.1.12 | Verified: honours `DEVECO_CONFIG_DIR` and loads this profile. `run --help` no longer lists `--dangerously-skip-permissions`; the flag is accepted and ignored, and `--auto` now exists. |

Install with `npm install -g opencode-ai@1.18.29` or
`npm install -g @deveco/deveco-code@0.1.12`. Other versions may work but are
not covered by the smoke test below. The DevEco explorer sends both `--auto`
(honoured by 0.1.12) and `--dangerously-skip-permissions` (honoured by
0.1.9), so a rule left at `ask` in a private profile is resolved on either
version. Set `--no-deveco-skip-permissions` to omit both flags.
The committed profiles additionally leave nothing at `ask`.

## Model wiring

The provider is the repository's LiteLLM proxy (`configs/litellm_proxy.yaml`),
read from the same variables the academic agents use:

| Variable | Used as |
| --- | --- |
| `ACADEMIC_API_BASE` | `provider.swe-explore.options.baseURL` (e.g. `http://127.0.0.1:4000/v1`) |
| `ACADEMIC_API_KEY` | `provider.swe-explore.options.apiKey` |

Only `{env:...}` references appear in the files; no secret is ever committed,
and the test suite rejects a profile that contains a literal key. The model
is `swe-explore/gpt-5.4` with a declared `limit` of 200k context and 64k
output tokens. Declare limits for any custom model: OpenCode resolves an
undeclared one to zero, which switches off overflow detection and proactive
compaction, so long explorations would hit the provider's hard limit
instead. Edit the `models` map, its `limit`, and the `model` key together to
evaluate another deployment.

The explorers redirect `HOME` to an empty temporary directory for every run,
so the user's own `~/.config/opencode` and `~/.config/deveco` are never read.
This also hides DevEco Code's `deveco auth login` state, which is why the
profile must carry its own provider instead of relying on the free
Huawei-account model.

## Read-only permissions

The agent is asked to explore, not to edit, and the profile enforces it:

- `permission.edit: deny` blocks `edit`, `write` and `patch`.
- `permission.doom_loop: allow`. `deny` here does not refuse the third
  identical tool call, it ends the whole session with an error, and an
  explorer that greps the same term three times is normal. The result would
  be a silent empty answer for that instance.
- `permission.bash: deny`. No shell allow-list is safe: OpenCode matches the
  outer command only, and even "read-only" programs run other programs
  (`rg --pre <cmd>`, `git grep -O<cmd>`, `find -exec`) or write files
  (`git log --output=`). The native `read`, `grep`, `glob` and `list` tools
  cover every legitimate exploration need.
- MCP tools are checked under their own names, so `permission.edit` does not
  reach them. Serena's editing, shell and memory tools are switched off under
  `tools` by glob (`serena_replace*`, `serena_insert*`, ...), and the seeded
  project file sets `read_only: true` and excludes the same tools by name.
  Either of those two is sufficient on its own. Serena's `planning` mode is
  also passed, but it is only a third, partial layer: it keeps
  `rename_symbol`, `replace_in_files` and the memory writers.
- `webfetch`, `websearch`, `external_directory`, `skill` and `question` are
  denied, so the run cannot leave the checkout or wait for a human.
- `task` (subagent delegation) is denied. OpenCode's `--format json` stream
  carries a subagent's final text but not its tool calls, so a delegated run
  would hide which tools did the work and make the MCP comparison below
  unreadable.
- `read`, `list`, `glob`, `grep`, `lsp` and `todowrite` are allowed.
- Every denied native tool (`write`, `edit`, `patch`, `multiedit`, `bash`,
  `task`) is also switched off under `tools`, so the model never spends a
  turn on a call that would be refused.
- `share: disabled`, `autoupdate: false` and `snapshot: false` keep the run
  offline apart from the model endpoint and stop the CLI writing snapshots.

OpenCode's `--auto` only resolves rules that would otherwise ask; explicit
`deny` rules stay in force.

## Semantic navigation over MCP

The `arkts` variant enables [Serena](https://github.com/oraios/serena), an
LSP-backed code-navigation MCP server, launched as
`serena start-mcp-server --context ide --mode planning --mode no-onboarding
--enable-web-dashboard false --enable-gui-log-window false
--project-from-cwd`. Install it with `uv tool install serena-agent` before
using that variant. The two `false` flags matter because the explorer gives
every run a fresh `HOME`: Serena's default config would otherwise open a
dashboard browser tab and a log window per benchmark instance.

`--project-from-cwd` activates the nearest ancestor of the checkout that
holds `.serena/project.yml` or `.git`. Benchmark snapshots are fetched
without `.git`, and `repos/` sits inside this repository, so a checkout
without its own project file would silently activate SWE-Explore-Bench
itself and every navigation call would return nothing. The MCP-on profiles
therefore carry `checkout/.serena/project.yml`, which the explorer places
into the checkout for each run (see the top of this file). It names the
TypeScript server, sets `read_only: true`, and excludes the editing tools.
Serena writes its language-server cache under `.serena/` in the checkout,
which the tests exclude from the "unchanged" check.

Serena installs language servers under its home directory, and the explorer
gives every run a fresh `HOME`. The profiles pass
`SERENA_HOME={env:SWE_EXPLORE_SERENA_HOME}` to the server so that cache can
persist: export `SWE_EXPLORE_SERENA_HOME` to a directory of your choice
before a benchmark run, or Serena reinstalls the TypeScript server for every
instance. The smoke test sets it to one directory for the whole session.

Serena has no ArkTS language server; `.ets` files fall outside its symbol
index, which is exactly the question the comparison below answers. Swap the
`mcp.serena` entry for another server to test a different navigation tool
and keep the `arkts-no-mcp` variant as the control.

## Running the benchmark

```bash
export ACADEMIC_API_BASE=http://127.0.0.1:4000/v1
export ACADEMIC_API_KEY=...
export SWE_EXPLORE_SERENA_HOME=~/.cache/swe-explore-serena

uv run python eval_runner.py \
  --bench bench.arkts.jsonl --repos repos \
  --explorers opencode \
  --opencode-config-dir configs/cli_agents/opencode/arkts \
  --top-k 5 --output "results/{explorer}/top{k}.jsonl"

uv run python eval_runner.py \
  --bench bench.arkts.jsonl --repos repos \
  --explorers deveco \
  --deveco-config-dir configs/cli_agents/deveco/arkts \
  --top-k 5 --output "results/{explorer}/top{k}.jsonl"
```

Use the `arkts-no-mcp` directories for the control arm.

## Tests

`tests/test_cli_agent_profiles.py` checks the committed files: valid JSON,
no literal credentials, bash denied, editing tools off for both native and
Serena tools, no `ask` rules, and the two variants differing only in the MCP
flag.

`tests/test_cli_agent_arkts_e2e.py` drives both explorers end to end against
the handwritten OpenHarmony fixture in `tests/fixtures/arkts_app`.

The agent's checkout is prepared as the
[fixture README](../../tests/fixtures/arkts_app/README.md) requires, plus the
Serena project file, exactly as a benchmark repository would be.

- **Scripted driver, always on.** `tests/scripted_cli_agent.py` runs as a
  real subprocess in place of the CLI. It reads the prompt from stdin, loads
  the profile from the config-dir variable of the CLI named in
  `SCRIPTED_AGENT_CLI` (so a stray variable for the other CLI cannot hijack
  a run), refuses profiles that are not read-only, lists the `.ets` files,
  follows the `import` from `Index.ets` to `DataSource.ets`, and answers with
  OpenCode-shaped JSONL events. The test asserts the exact file and line range
  from `expected_locations.json`, that the checkout is byte-identical
  afterwards, and that token usage from `step_finish` events was collected.
- **Real driver, opt-in.** Skipped unless `SWE_EXPLORE_CLI_SMOKE=1`, and the
  MCP-on variant is also skipped when `serena` is not on `PATH`, because
  OpenCode would otherwise continue without it and the two arms would be
  identical. It runs the installed binaries over both variants, asserts that
  some returned region overlaps the expected one, that the MCP-on arm made at
  least one Serena *navigation* call that completed (failed calls and
  administrative calls such as reading Serena's configuration are counted
  separately, since neither shows that semantic navigation did any work) and
  the MCP-off arm none, and prints one `[cli-smoke]` line per run with
  exact-hit, region width, rank, tool-call count, the three MCP counts and
  tokens, so the arms can be compared:

```bash
SWE_EXPLORE_CLI_SMOKE=1 \
ACADEMIC_API_BASE=http://127.0.0.1:4000/v1 ACADEMIC_API_KEY=... \
uv run --with pytest python -m pytest tests/test_cli_agent_arkts_e2e.py -k real -s
```

| Variable | Effect |
| --- | --- |
| `SWE_EXPLORE_CLI_SMOKE=1` | Enables the real-CLI tests; otherwise they are skipped. |
| `SWE_EXPLORE_OPENCODE_BIN`, `SWE_EXPLORE_DEVECO_BIN` | Binary to run instead of the one on `PATH`. |
| `SWE_EXPLORE_OPENCODE_TIMEOUT`, `SWE_EXPLORE_DEVECO_TIMEOUT` | Per-run limit in seconds (default 600). |
| `SWE_EXPLORE_SERENA_HOME` | Persistent Serena home for the MCP-on arm; the test picks a session directory when unset. |

A missing binary, proxy variable or `serena` skips the affected test rather
than failing it.
