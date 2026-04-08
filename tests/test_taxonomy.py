# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utils/taxonomy.py — semantic label dictionaries and loading."""

import os
import tempfile

import pytest

from utils.taxonomy import (
    BOXY_SEM2NAME,
    SSI_COLORS,
    SSI_COLORS_ALT,
    SSI_NAME2SEM,
    SSI_SEM2NAME,
    TEXT2COLORS,
    load_text_labels,
)


class TestSemDicts:
    def test_boxy_inverse(self):
        """Verify name->id inverse can be built without collisions."""
        name2sem = {val: key for key, val in BOXY_SEM2NAME.items()}
        for sem_id, name in BOXY_SEM2NAME.items():
            assert name2sem[name] == sem_id

    def test_ssi_bijectivity(self):
        for sem_id, name in SSI_SEM2NAME.items():
            assert SSI_NAME2SEM[name] == sem_id

    def test_boxy_has_invalid_and_unknown(self):
        assert -1 in BOXY_SEM2NAME
        assert 0 in BOXY_SEM2NAME

    def test_ssi_superset_of_boxy(self):
        """SSI should have all BOXY categories (plus extras like Ceiling)."""
        for name in BOXY_SEM2NAME.values():
            assert name in SSI_SEM2NAME.values()


class TestColorDicts:
    def test_ssi_colors_valid_rgb(self):
        for name, rgb in SSI_COLORS.items():
            assert len(rgb) == 3, f"{name} has {len(rgb)} channels"
            for c in rgb:
                assert 0.0 <= c <= 1.0, f"{name} channel {c} out of range"

    def test_ssi_colors_alt_valid_rgb(self):
        for name, rgb in SSI_COLORS_ALT.items():
            for c in rgb:
                assert 0.0 <= c <= 1.0

    def test_ssi_colors_cover_all_categories(self):
        """Every SSI category should have a color."""
        for name in SSI_SEM2NAME.values():
            assert name in SSI_COLORS, f"Missing color for {name}"

    def test_text2colors_lowercase_keys(self):
        for key in TEXT2COLORS:
            assert key == key.lower()


class TestLoadTextLabels:
    def test_custom_list_passthrough(self):
        """When given a list of strings that don't match a file, return as-is."""
        labels = ["cat", "dog", "bird"]
        result = load_text_labels(labels)
        assert result == labels

    def test_unknown_label_set(self):
        """Unknown label set name should return the name as a list."""
        result = load_text_labels("nonexistent_label_set_xyz")
        assert result == ["nonexistent_label_set_xyz"]

    def test_none_input(self):
        result = load_text_labels(None)
        assert isinstance(result, list)

    def test_real_file(self):
        """If a real label file exists, verify it loads."""
        tmpdir = tempfile.mkdtemp()
        label_file = os.path.join(tmpdir, "test_classes.csv")
        with open(label_file, "w") as f:
            f.write("apple\nbanana\ncherry\n")

        import utils.taxonomy as tax

        old_dir = tax._LABELS_DIR
        tax._LABELS_DIR = tmpdir
        try:
            result = load_text_labels("test")
            assert result == ["apple", "banana", "cherry"]
        finally:
            tax._LABELS_DIR = old_dir
            os.remove(label_file)
            os.rmdir(tmpdir)

    def test_empty_file_raises(self):
        tmpdir = tempfile.mkdtemp()
        label_file = os.path.join(tmpdir, "empty_classes.csv")
        with open(label_file, "w") as f:
            f.write("")

        import utils.taxonomy as tax

        old_dir = tax._LABELS_DIR
        tax._LABELS_DIR = tmpdir
        try:
            with pytest.raises(ValueError, match="empty"):
                load_text_labels("empty")
        finally:
            tax._LABELS_DIR = old_dir
            os.remove(label_file)
            os.rmdir(tmpdir)
