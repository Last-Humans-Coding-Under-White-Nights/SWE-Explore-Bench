from eval_runner import _resolve_repo_dir


def test_resolve_repo_dir_accepts_workspace_or_checkout_root(tmp_path):
    repo = tmp_path / "repos" / "org__repo-1"
    repo.mkdir(parents=True)

    from_workspace = _resolve_repo_dir(
        "repos/org__repo-1", tmp_path, "org__repo-1"
    )
    from_checkout_root = _resolve_repo_dir(
        "repos/org__repo-1", tmp_path / "repos", "org__repo-1"
    )

    assert from_workspace == repo
    assert from_checkout_root == repo
