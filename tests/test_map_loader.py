import tempfile
import unittest
from pathlib import Path

from examples.utilities.map_loader import MapValidationError, load_world, parse_world_text


class TestMapLoader(unittest.TestCase):
    """Covers the shared ASCII map loader used by both A* and PPO."""

    def test_parse_world_text_preserves_spaces_and_counts(self):
        world = "wwwww\nwa gw\nw o w\nwwwww\n"
        normalized, metadata = parse_world_text(world, min_starts=1, max_starts=1, min_goals=1, max_goals=1)
        self.assertEqual(normalized.splitlines()[1], "wa gw")
        self.assertEqual(metadata.width, 5)
        self.assertEqual(metadata.height, 4)
        self.assertEqual(metadata.starts, ((1, 1),))
        self.assertEqual(metadata.goals, ((3, 1),))
        self.assertEqual(metadata.holes, ((2, 2),))

    def test_parse_world_text_rejects_irregular_rows(self):
        with self.assertRaises(MapValidationError):
            parse_world_text("wwww\nwa gw\nwww\n", min_goals=1)

    def test_parse_world_text_rejects_invalid_symbols(self):
        with self.assertRaises(MapValidationError):
            parse_world_text("wwwww\nwa xw\nwwwww\n", min_goals=0)

    def test_load_world_rejects_duplicate_goals_when_max_is_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            world_path = Path(temp_dir) / "duplicate_goal.txt"
            world_path.write_text("wwwww\nwaggw\nwwwww\n", encoding="utf-8")
            with self.assertRaises(MapValidationError):
                load_world(world_path, min_starts=1, max_starts=1, min_goals=1, max_goals=1)
