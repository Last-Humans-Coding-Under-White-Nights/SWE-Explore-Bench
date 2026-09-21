"""The committed OpenCode / DevEco Code profiles must stay reproducible.

Anyone reproducing a published ArkTS number starts from these files, so
they must load, carry no secrets, and pin the agent to read-only tools.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from explorers._cli_agent_base import CHECKOUT_SEED_DIR
from explorers.deveco import DevEcoExplorer
from explorers.opencode import OpenCodeExplorer

PROFILES_ROOT = Path(__file__).resolve().parents[1] / "configs" / "cli_agents"

#: cli -> config filename, taken from the explorer that will load it.
CLIS = {
    "opencode": OpenCodeExplorer.config_filename,
    "deveco": DevEcoExplorer.config_filename,
}
VARIANTS = ("arkts", "arkts-no-mcp")
PROFILES = [(cli, variant) for cli in CLIS for variant in VARIANTS]

#: A leaf under a key that contains one of these words (as a whole word of a
#: camelCase, snake_case or UPPER_CASE key, so "apiKey", "api_key", "API_KEY"
#: and "APIKey" match but "keyword" does not) must be an {env:...} reference.
SECRET_WORDS = frozenset({
    "key", "keys", "secret", "secrets", "token", "tokens", "password", "passwords",
    "authorization", "header", "headers", "credential", "credentials",
})
KEY_WORDS = re.compile(r"[a-z]+|\d+|[A-Z]+(?![a-z])|[A-Z][a-z]*")
ENV_REF = re.compile(r"^\{env:[A-Z0-9_]+\}$")
#: Shapes a pasted secret takes, checked on every string value whatever its key.
SECRET_VALUE = re.compile(
    r"(?i)bearer\s+\S{8,}|\bsk-[A-Za-z0-9_-]{8,}|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\b[A-Fa-f0-9]{32,}\b|(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9+/=_-])"
)

# The permission policy and its reasons live in configs/cli_agents/README.md,
# "Read-only permissions"; these tuples only pin what that section states.
READ_ONLY_TOOLS = ("read", "list", "glob", "grep")
DENIED_TOOLS = ("edit", "bash", "task", "webfetch", "websearch",
                "external_directory", "skill", "question")
DISABLED_TOOLS = (
    "write", "edit", "patch", "multiedit", "bash", "task",
    "serena_create_text_file", "serena_replace*", "serena_insert*", "serena_delete*",
    "serena_rename_symbol", "serena_execute_shell_command",
    "serena_write_memory", "serena_edit_memory", "serena_remove_project",
)


def load(cli: str, variant: str) -> dict:
    path = PROFILES_ROOT / cli / variant / CLIS[cli]
    assert path.is_file(), f"profile missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def leaves(node, path=()):
    """Yield (key path, value) for every leaf of a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, path + (index,))
    else:
        yield path, node


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_profile_carries_no_literal_credentials(cli, variant):
    for key_path, value in leaves(load(cli, variant)):
        words = {w.lower() for key in key_path for w in KEY_WORDS.findall(str(key))}
        # Numeric token budgets are settings; authentication fields must still
        # use env references, even when nested under a limits section.
        leaf_words = {w.lower() for w in KEY_WORDS.findall(str(key_path[-1]))}
        token_budget = (
            words & SECRET_WORDS <= {"token", "tokens"}
            and leaf_words in (
                {"max", "tokens"}, {"input", "tokens"}, {"output", "tokens"},
                {"reserved", "tokens"}, {"token", "limit"}, {"token", "count"},
                {"num", "tokens"},
            )
            and isinstance(value, int) and not isinstance(value, bool)
        )
        if words & SECRET_WORDS and not token_budget:
            location = "/".join(map(str, key_path))
            assert isinstance(value, str) and ENV_REF.match(value), (
                f"{location} must be an {{env:VAR}} reference, got {value!r}"
            )
        if isinstance(value, str) and not ENV_REF.match(value):
            assert not SECRET_VALUE.search(value), (
                f"{'/'.join(map(str, key_path))} looks like a pasted secret"
            )


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_profile_is_explicitly_read_only(cli, variant):
    permission = load(cli, variant)["permission"]
    for tool in DENIED_TOOLS:
        assert permission[tool] == "deny", tool
    for tool in READ_ONLY_TOOLS:
        assert permission[tool] == "allow", tool
    # "deny" would abort the whole session on the third identical tool call
    # instead of refusing that call, and repeated greps are normal exploration.
    assert permission["doom_loop"] == "allow"


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_profile_hides_denied_tools_from_the_model(cli, variant):
    profile = load(cli, variant)
    for tool in DISABLED_TOOLS:
        assert profile["tools"][tool] is False, tool
    command = profile["mcp"]["serena"]["command"]
    assert command[:2] == ["serena", "start-mcp-server"]
    assert "planning" in command and "no-onboarding" in command
    assert "--project-from-cwd" in command


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_serena_keeps_its_language_server_cache_across_runs(cli, variant):
    """Every run gets a fresh HOME, so without this Serena would reinstall
    the language server per benchmark instance."""
    environment = load(cli, variant)["mcp"]["serena"]["environment"]
    assert environment["SERENA_HOME"] == "{env:SWE_EXPLORE_SERENA_HOME}"


@pytest.mark.parametrize("cli", list(CLIS))
def test_mcp_variant_seeds_the_serena_project_file(cli):
    """--project-from-cwd walks up the ancestors, so a checkout without its
    own project file would silently activate this repository instead."""
    seed = PROFILES_ROOT / cli / "arkts" / CHECKOUT_SEED_DIR / ".serena" / "project.yml"
    assert seed.is_file()
    text = seed.read_text(encoding="utf-8")
    assert "read_only: true" in text
    for tool in ("replace_in_files", "rename_symbol", "write_memory"):
        assert f"- {tool}" in text
    assert not (PROFILES_ROOT / cli / "arkts-no-mcp" / CHECKOUT_SEED_DIR).exists()


def test_both_mcp_variants_seed_the_same_project_file():
    files = [
        (PROFILES_ROOT / cli / "arkts" / CHECKOUT_SEED_DIR / ".serena" / "project.yml")
        .read_bytes() for cli in CLIS
    ]
    assert files[0] == files[1]


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_serena_runs_headless(cli, variant):
    """A fresh HOME resets Serena's config, whose defaults open a dashboard
    browser tab and a log window on every launch."""
    command = load(cli, variant)["mcp"]["serena"]["command"]
    for flag in ("--enable-web-dashboard", "--enable-gui-log-window"):
        assert flag in command, f"{flag} missing from serena command"
        assert command[command.index(flag) + 1:command.index(flag) + 2] == ["false"], flag


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_profile_never_asks(cli, variant):
    """A headless run must not stall on an approval prompt."""
    for key_path, value in leaves(load(cli, variant)["permission"]):
        assert value != "ask", "/".join(map(str, key_path))


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_profile_routes_the_model_through_the_proxy_env(cli, variant):
    profile = load(cli, variant)
    provider_id, model_id = profile["model"].split("/", 1)
    provider = profile["provider"][provider_id]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "{env:ACADEMIC_API_BASE}"
    assert provider["options"]["apiKey"] == "{env:ACADEMIC_API_KEY}"
    assert model_id in provider["models"]
    # A custom model resolves to zero limits unless declared, and a zero
    # context limit switches off OpenCode's overflow detection and compaction.
    limit = provider["models"][model_id]["limit"]
    assert limit["context"] > 0 and limit["output"] > 0


@pytest.mark.parametrize("cli,variant", PROFILES)
def test_profile_stays_offline_apart_from_the_model_endpoint(cli, variant):
    profile = load(cli, variant)
    assert profile["share"] == "disabled"
    assert profile["autoupdate"] is False
    assert profile["snapshot"] is False


@pytest.mark.parametrize("cli", list(CLIS))
def test_variants_differ_only_in_mcp_enabled(cli):
    with_mcp = load(cli, "arkts")
    without_mcp = load(cli, "arkts-no-mcp")
    assert with_mcp["mcp"]["serena"]["enabled"] is True
    assert without_mcp["mcp"]["serena"]["enabled"] is False
    with_mcp["mcp"]["serena"]["enabled"] = False
    assert with_mcp == without_mcp


@pytest.mark.parametrize("variant", VARIANTS)
def test_opencode_and_deveco_profiles_share_one_shape(variant):
    """DevEco Code is an OpenCode fork with the same config schema."""
    assert load("opencode", variant) == load("deveco", variant)
