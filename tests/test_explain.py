import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explain import GROUPS, LABELS  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402


def test_every_feature_has_a_plain_language_label():
    assert set(LABELS) == set(FEATURE_NAMES)


def test_every_feature_belongs_to_exactly_one_theme():
    grouped = [f for feats in GROUPS.values() for f in feats]
    assert sorted(grouped) == sorted(FEATURE_NAMES)
