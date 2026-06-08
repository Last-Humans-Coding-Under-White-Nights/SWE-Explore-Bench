"""Compatibility shim for LocAgent + llama-index-core >= 0.12."""
import os, sys

locagent_root = os.environ.get('LOCAGENT_ROOT', os.getcwd())
if locagent_root not in sys.path:
    sys.path.insert(0, locagent_root)
os.chdir(locagent_root)

# Patch 1: llama-index alias
import llama_index.core.vector_stores.simple as _vs_mod
if not hasattr(_vs_mod, '_build_metadata_filter_fn') and hasattr(_vs_mod, 'build_metadata_filter_fn'):
    _vs_mod._build_metadata_filter_fn = _vs_mod.build_metadata_filter_fn

# Patch 2: load_dataset for local JSONL
import datasets as _ds_mod
if not hasattr(_ds_mod, '_shim_patched'):
    _orig_load = _ds_mod.load_dataset
    def _patched_load(path, *a, **kw):
        if isinstance(path, str) and os.path.isfile(path) and path.endswith(('.jsonl', '.json')):
            return _orig_load('json', *a, data_files=path, **kw)
        return _orig_load(path, *a, **kw)
    _ds_mod.load_dataset = _patched_load
    _ds_mod._shim_patched = True

# Patch 3: Force fork in torch.multiprocessing.spawn submodule
import torch.multiprocessing  # triggers import of submodule
_spawn_mod = sys.modules['torch.multiprocessing.spawn']
_orig_start = _spawn_mod.start_processes
def _fork_start(*a, **kw):
    kw['start_method'] = 'fork'
    return _orig_start(*a, **kw)
_spawn_mod.start_processes = _fork_start

# Patch 4: Extend argparse model whitelist to accept any model
import argparse
_orig_add_argument = argparse.ArgumentParser.add_argument
def _patched_add_argument(self, *args, **kwargs):
    if args and args[0] == "--model" and "choices" in kwargs:
        kwargs.pop("choices")  # remove whitelist restriction
    return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _patched_add_argument

# Patch 5: Extend cost table to accept unknown models (default to gpt-4o pricing)
sys.path.insert(0, locagent_root)
from util.cost_analysis import MODEL_COST_PER_INPUT, MODEL_COST_PER_OUTPUT
class _CostDefault(dict):
    def __missing__(self, key):
        return 2.5e-06  # default to gpt-4o pricing
_input_defaults = _CostDefault(MODEL_COST_PER_INPUT)
_output_defaults = _CostDefault(MODEL_COST_PER_OUTPUT)
import util.cost_analysis as _ca
_ca.MODEL_COST_PER_INPUT = _input_defaults
_ca.MODEL_COST_PER_OUTPUT = _output_defaults

# Patch 6: Use local repo via symlink instead of GitHub cloning
_local_repo = os.environ.get('LOCAL_REPO_PATH', '')
if _local_repo:
    import shutil
    import util.benchmark.git_repo_manager as _grm

    _orig_maybe_clone = _grm.maybe_clone
    def _patched_maybe_clone(repo_url, repo_dir):
        """Create symlink to local repo instead of cloning from GitHub."""
        if not os.path.exists(os.path.join(repo_dir, '.git')):
            os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir)
            os.symlink(_local_repo, repo_dir)
        # else: already exists, skip
    _grm.maybe_clone = _patched_maybe_clone

    _orig_checkout = _grm.checkout_commit
    def _patched_checkout(repo_dir, commit_hash):
        """Skip checkout for local repos (already at correct state)."""
        pass
    _grm.checkout_commit = _patched_checkout

    _orig_clean = _grm.clean_and_reset_state
    def _patched_clean(repo_dir):
        """Skip clean for local repos (don't modify original)."""
        pass
    _grm.clean_and_reset_state = _patched_clean

# Patch 7: Fix BM25Retriever crash when EpicSplitter produces empty nodes.
# EpicSplitter.__init__() fails on all files, treesitter fallback returns
# empty list, and BM25Retriever.from_defaults(nodes=[]) raises ValueError.
# Fix: return a no-op retriever so LocAgent falls back to entity search.
from llama_index.retrievers.bm25 import BM25Retriever as _BM25Cls

class _EmptyBM25Retriever:
    """Mock retriever that returns no results when BM25 index is empty."""
    def retrieve(self, query):
        return []
    def persist(self, path):
        pass

_orig_bm25_defaults = _BM25Cls.from_defaults

@classmethod
def _safe_bm25_defaults(cls, index=None, nodes=None, docstore=None, **kwargs):
    if nodes is not None and len(nodes) == 0:
        return _EmptyBM25Retriever()
    return _orig_bm25_defaults.__func__(cls, index=index, nodes=nodes, docstore=docstore, **kwargs)

_BM25Cls.from_defaults = _safe_bm25_defaults

import runpy
runpy.run_path(os.path.join(locagent_root, 'auto_search_main.py'), run_name='__main__')
