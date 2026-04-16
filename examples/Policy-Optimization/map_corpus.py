"""Helpers for locating and loading the committed map corpus from disk."""

from pathlib import Path

from utilities.map_loader import load_world


class LoadedMap:
    """Represents one validated map file plus its parsed metadata."""

    def __init__(self, path, world_text, metadata):
        self.path = Path(path)
        self.world_text = world_text
        self.metadata = metadata
        self.rows = tuple(world_text.splitlines())


def stage_directory(maps_root, split, stage):
    """Returns the directory that contains one split and stage worth of map files."""

    return Path(maps_root) / split / f"stage{stage}"


def list_stage_map_files(maps_root, split, stage):
    """Lists all map files for a given split and stage in deterministic order."""

    stage_dir = stage_directory(maps_root, split, stage)
    files = sorted(stage_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No map files found in {stage_dir}.")
    return files


def load_policy_map(path):
    """Loads one policy-optimization map and enforces a single start and goal."""

    world_text, metadata = load_world(path, min_starts=1, max_starts=1, min_goals=1, max_goals=1)
    return LoadedMap(path, world_text, metadata)


def load_stage_maps(maps_root, split, stage):
    """Loads every validated map for a stage and split into memory."""

    return [load_policy_map(path) for path in list_stage_map_files(maps_root, split, stage)]


def ensure_map_paths(maps):
    """Normalizes a mixed list of paths and preloaded maps into LoadedMap objects."""

    resolved = []
    for item in maps:
        if isinstance(item, LoadedMap):
            resolved.append(item)
        else:
            resolved.append(load_policy_map(item))
    if not resolved:
        raise ValueError("At least one map must be provided.")
    return resolved
