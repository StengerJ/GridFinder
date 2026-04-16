"""Utilities for reading rectangular ASCII map files from disk."""

from collections import Counter
from pathlib import Path


VALID_MAP_SYMBOLS = frozenset({"w", "a", "g", "o", " "})


class MapValidationError(ValueError):
    """Raised when a map file does not match the expected ASCII format."""


class MapMetadata:
    """Stores the parsed positions and counts for one loaded map."""

    def __init__(self, width, height, starts, goals, holes, walls, empty, symbol_counts):
        self.width = width
        self.height = height
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.holes = tuple(holes)
        self.walls = tuple(walls)
        self.empty = tuple(empty)
        self.symbol_counts = dict(symbol_counts)


def _normalize_lines(text: str) -> list[str]:
    """Normalizes line endings and removes only the final trailing newline."""

    if not text:
        raise MapValidationError("Map text is empty.")

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        raise MapValidationError("Map text is empty.")
    return lines


def parse_world_text(
    text: str,
    *,
    min_starts: int = 1,
    max_starts: int | None = 1,
    min_goals: int = 1,
    max_goals: int | None = None,
) -> tuple[str, MapMetadata]:
    """Validates ASCII world text and returns normalized text plus parsed metadata."""

    lines = _normalize_lines(text)
    width = len(lines[0])
    if width == 0:
        raise MapValidationError("Map rows must not be empty.")

    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    holes: list[tuple[int, int]] = []
    walls: list[tuple[int, int]] = []
    empty: list[tuple[int, int]] = []
    counts: Counter[str] = Counter()

    for row_index, row in enumerate(lines):
        if len(row) != width:
            raise MapValidationError("Map rows must all have the same width.")
        for col_index, symbol in enumerate(row):
            if symbol not in VALID_MAP_SYMBOLS:
                raise MapValidationError(f"Unsupported map symbol {symbol!r}.")
            counts[symbol] += 1
            coord = (col_index, row_index)
            if symbol == "a":
                starts.append(coord)
            elif symbol == "g":
                goals.append(coord)
            elif symbol == "o":
                holes.append(coord)
            elif symbol == "w":
                walls.append(coord)
            else:
                empty.append(coord)

    if len(starts) < min_starts:
        raise MapValidationError(f"Expected at least {min_starts} start cell(s).")
    if max_starts is not None and len(starts) > max_starts:
        raise MapValidationError(f"Expected at most {max_starts} start cell(s).")
    if len(goals) < min_goals:
        raise MapValidationError(f"Expected at least {min_goals} goal cell(s).")
    if max_goals is not None and len(goals) > max_goals:
        raise MapValidationError(f"Expected at most {max_goals} goal cell(s).")

    metadata = MapMetadata(
        width=width,
        height=len(lines),
        starts=tuple(starts),
        goals=tuple(goals),
        holes=tuple(holes),
        walls=tuple(walls),
        empty=tuple(empty),
        symbol_counts=dict(counts),
    )
    return "\n".join(lines), metadata


def load_world(
    path: str | Path,
    *,
    min_starts: int = 1,
    max_starts: int | None = 1,
    min_goals: int = 1,
    max_goals: int | None = None,
) -> tuple[str, MapMetadata]:
    """Loads a map file from disk and validates it with the requested symbol counts."""

    world_path = Path(path)
    world_text = world_path.read_text(encoding="utf-8")
    return parse_world_text(
        world_text,
        min_starts=min_starts,
        max_starts=max_starts,
        min_goals=min_goals,
        max_goals=max_goals,
    )
