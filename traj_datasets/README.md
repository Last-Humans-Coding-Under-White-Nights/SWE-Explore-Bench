# traj_datasets Module

The traj_datasets module provides tools for converting various trajectory formats to a unified format for swe-explore.

## Unified Format

The unified trajectory format is defined as:

```python
{
    "info": {
        "repo": str,           # Repository name
        "model": str,          # Model name
        "issue": str,          # Issue/task description
        "answer": str,         # Model's solution/patch
        "instance_id": str,    # Unique instance identifier
        "rounds": int,         # Number of interaction rounds
        # ... other metadata
    },
    "traj": [
        {
            "role": "system" | "assistant" | "user" | "tool",
            "content": str | list | None,
            # Optional fields based on role:
            "tool_calls": [...],      # For assistant messages
            "name": str,              # For tool messages
            "tool_call_id": str,      # For tool messages
            "meta": {...}             # Additional metadata
        },
        ...
    ]
}
```

## Components

### 1. Pydantic Models (`models.py`)

Defines the data models for the unified format:
- `TrajectoryInfo`: Metadata about the trajectory
- `TrajectoryMessage`: Individual messages in the trajectory
- `ToolCall`, `ToolCallFunction`: Tool call structures
- `UnifiedTrajectory`: Complete trajectory with info and messages

### 2. Nebius Dataset Loader (`nebius_loader.py`)

Loads and converts the `nebius/SWE-rebench-openhands-trajectories` dataset to unified format.

**Features:**
- Type-safe conversion with full type annotations
- Rich console output for progress tracking
- Proper error handling with detailed tracebacks
- Automatic JSON deserialization for tool call arguments
- Optional filtering support
- Easy-to-use CLI with typer (--help for all options)

**Usage:**

```python
from traj_datasets import load_nebius_dataset

# Load entire dataset
trajectories = load_nebius_dataset(split="train")

# Load subset
trajectories = load_nebius_dataset(split="train[:10]")

# Save to files with rich progress output
trajectories = load_nebius_dataset(
    split="train",
    output_dir="converted_nebius"
)

# With custom filter
trajectories = load_nebius_dataset(
    split="train",
    filter_fn=lambda row: row["status"] == "success"
)
```

**Command line:**

```bash
# Load first 5 trajectories (default)
python -m traj_datasets.nebius_loader

# Load entire train split
python -m traj_datasets.nebius_loader --split train

# Load and save to files
python -m traj_datasets.nebius_loader --split train --output-dir ./nebius_converted

# Load first 100 without showing details
python -m traj_datasets.nebius_loader --split "train[:100]" --no-show-first

# Show help
python -m traj_datasets.nebius_loader --help
```

### 3. Existing Trajectory Adapter (`existing_adapter.py`)

Converts existing trajectory format (from trajs/ directory) to unified format.

**Features:**
- Clean, modular code with helper functions
- Comprehensive type annotations
- Rich console output with colored status indicators
- Detailed error reporting with `traceback.print_exc()`
- Proper exception handling (raises meaningful errors instead of silently catching)
- Batch conversion with summary statistics
- Easy-to-use CLI with typer (--help for all options)

**Usage:**

```python
from traj_datasets import convert_existing_trajectory
from pathlib import Path

# Convert single file (raises exception on error)
traj_file = Path("trajs/trajs-mini-claude4.5-pruner/pytest-dev__pytest-5809/pytest-dev__pytest-5809.traj.json")
unified_traj = convert_existing_trajectory(traj_file)

# Convert entire directory with rich output
from traj_datasets.existing_adapter import convert_existing_trajectory_dir

trajectories = convert_existing_trajectory_dir(
    input_dir=Path("trajs/trajs-mini-claude4.5-pruner"),
    output_dir=Path("converted_trajectories")
)
# Outputs:
# ✓ Converted: file1.traj.json
# ✓ Converted: file2.traj.json
# ✗ Failed: file3.traj.json
# 
# Successfully converted: 2
# Failed: 1
```

**Command line:**

```bash
# Convert single file (outputs to .unified.json by default)
python -m traj_datasets.existing_adapter path/to/trajectory.traj.json

# Convert single file with custom output
python -m traj_datasets.existing_adapter path/to/trajectory.traj.json --output custom.json

# Convert directory (outputs to <dirname>_unified/ by default)
python -m traj_datasets.existing_adapter path/to/trajs/directory

# Convert directory with custom output
python -m traj_datasets.existing_adapter path/to/trajs/directory --output-dir ./converted

# Show help
python -m traj_datasets.existing_adapter --help
```

## Integration with bench_build.py

The `bench_build.py` script has been updated to support both old and unified formats:

1. **Automatic format detection**: The script automatically detects whether a trajectory file is in old or unified format
2. **Backward compatibility**: Old format files continue to work without changes
3. **Unified format support**: New unified format files are automatically converted internally

**Usage remains the same:**

```bash
# Build benchmark from trajectories
python bench_build.py build --trajs-dir trajs --output bench.jsonl

# Filter by model
python bench_build.py build --trajs-dir trajs --model trajs-mini-claude4.5-pruner

# Filter by instance
python bench_build.py build --trajs-dir trajs --instance-filter "django__django-"
```

## Examples

See `examples.py` for comprehensive usage examples:

```bash
python -m traj_datasets.examples
```

## Installation

The traj_datasets module requires additional dependencies:

```bash
# Install with uv (recommended)
uv sync

# Or manually add dependencies
uv add pydantic datasets rich

# Or with pip
pip install pydantic datasets rich
```

## Field Mappings

### Nebius Dataset → Unified Format

| Nebius Field | Unified Field | Notes |
|--------------|---------------|-------|
| `instance_id` | `info.instance_id` | Direct mapping |
| `instruction` | `info.issue` | Task description |
| `git_patch` | `info.answer` | Model's solution |
| `trajectory` | `traj` | Message list |
| `metadata.model_name` | `info.model` | Model identifier |
| `status` | `info.exit_status` | Execution status |

### Existing Format → Unified Format

| Old Field | Unified Field | Notes |
|-----------|---------------|-------|
| `info` | `info` | Preserved with additions |
| `messages` | `traj` | Message list |
| `info.submission` | `info.answer` | Model's patch |
| `info.exit_status` | `info.exit_status` | Direct mapping |
| `info.config` | `info.config` | Configuration metadata |

## Role-Based Message Fields

Different message roles have different required/optional fields:

- **system**: `role`, `content`
- **assistant**: `role`, `content`, `tool_calls` (optional)
- **user**: `role`, `content`
- **tool**: `role`, `content`, `name`, `tool_call_id`

## Code Quality

This module follows strict code quality standards:

1. **Type Safety**: Full type annotations on all functions and variables
2. **Error Handling**: Proper exceptions instead of silent try-catch blocks
3. **Logging**: Uses `rich` library for beautiful console output
4. **Debugging**: Uses `traceback.print_exc()` for detailed error tracing
5. **Modularity**: Clean, single-responsibility functions
6. **Modern Python**: Leverages Python 3.10+ features (union types, walrus operator)
7. **Performance**: Lazy imports to avoid loading heavy dependencies (like HuggingFace `datasets`) unless actually needed

## Performance

The module uses **lazy imports** to optimize startup time:
- Heavy dependencies (like HuggingFace `datasets`) are only loaded when actually used
- Running `existing_adapter` won't load the `datasets` library
- Running `nebius_loader` will load the `datasets` library on first use
- This makes CLI commands start instantly instead of waiting 3-5 seconds for imports

## Notes

1. The unified format uses Pydantic for validation, ensuring data consistency
2. All converters preserve additional metadata in the `meta` field
3. Tool call arguments are automatically deserialized from JSON strings
4. The format is extensible - additional fields are allowed via `extra = "allow"`
5. Failed conversions are reported with detailed tracebacks, not silently ignored

