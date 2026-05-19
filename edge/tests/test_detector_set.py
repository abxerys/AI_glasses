"""Tests for ItemDetectorSet — routes a target class to the right detector.

We mock the detectors with simple stand-ins so the tests don't need real
.pt weights or ultralytics.
"""

from dataclasses import dataclass

from edge.vision.detector import ItemDetectorSet


@dataclass
class _FakeDetector:
    names: dict[int, str]


def test_empty_set():
    s = ItemDetectorSet([])
    assert s.empty
    assert s.for_class("anything") is None
    assert s.all_classes() == set()


def test_first_match_wins():
    shopping = _FakeDetector(names={0: "AD_milk", 1: "Red_Bull"})
    coco = _FakeDetector(names={0: "cell phone", 1: "bottle", 2: "chair"})
    s = ItemDetectorSet([shopping, coco])

    assert s.for_class("Red_Bull") is shopping
    assert s.for_class("cell phone") is coco
    assert s.for_class("bottle") is coco
    assert s.for_class("non_existent") is None


def test_shopping_preferred_on_overlap():
    # if a class name happened to appear in both, the first detector wins
    shopping = _FakeDetector(names={0: "bottle"})
    coco = _FakeDetector(names={0: "bottle"})
    s = ItemDetectorSet([shopping, coco])
    assert s.for_class("bottle") is shopping


def test_all_classes_union():
    a = _FakeDetector(names={0: "x", 1: "y"})
    b = _FakeDetector(names={0: "y", 1: "z"})
    s = ItemDetectorSet([a, b])
    assert s.all_classes() == {"x", "y", "z"}


def test_none_detectors_filtered():
    a = _FakeDetector(names={0: "x"})
    s = ItemDetectorSet([None, a, None])
    assert not s.empty
    assert len(s.detectors) == 1
    assert s.for_class("x") is a
