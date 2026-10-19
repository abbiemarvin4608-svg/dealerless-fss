"""
Tests for results I/O: merging per-model precision results without clobbering.

TDD: written before results_io.py exists.
"""

import pytest
import pandas as pd
from pathlib import Path

from results_io import merge_results


class TestMergeResults:
    """Test merging new model results into an existing CSV without data loss."""

    def test_merge_into_empty_creates_file(self, tmp_path):
        """Merging into a non-existent CSV should create it with the new rows."""
        csv_path = tmp_path / "e0.csv"
        new_rows = [
            {'model': 'MLP', 'ell': 'float32', 'accuracy': 97.5},
            {'model': 'MLP', 'ell': 20, 'accuracy': 97.2},
        ]

        merge_results(csv_path, new_rows)

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert set(df['model']) == {'MLP'}

    def test_merge_preserves_other_models(self, tmp_path):
        """
        CRITICAL: Merging ResNet results must NOT delete existing MLP results.

        This is the exact bug that motivated this module: to_csv() overwrite
        was destroying MLP data when the ResNet sweep ran.
        """
        csv_path = tmp_path / "e0.csv"
        # Existing MLP data
        merge_results(csv_path, [
            {'model': 'MLP', 'ell': 20, 'accuracy': 97.2},
            {'model': 'MLP', 'ell': 32, 'accuracy': 97.5},
        ])

        # Add ResNet data
        merge_results(csv_path, [
            {'model': 'ResNet20', 'ell': 20, 'accuracy': 88.0},
            {'model': 'ResNet20', 'ell': 32, 'accuracy': 91.0},
        ])

        df = pd.read_csv(csv_path)
        models = set(df['model'])
        assert 'MLP' in models, "MLP data was clobbered!"
        assert 'ResNet20' in models, "ResNet20 data missing!"
        assert len(df) == 4

    def test_merge_replaces_same_model(self, tmp_path):
        """Re-running the same model should replace its rows, not duplicate."""
        csv_path = tmp_path / "e0.csv"
        merge_results(csv_path, [
            {'model': 'MLP', 'ell': 20, 'accuracy': 90.0},  # old (bad) value
        ])

        # Re-run MLP with corrected value
        merge_results(csv_path, [
            {'model': 'MLP', 'ell': 20, 'accuracy': 97.2},  # new (correct) value
        ])

        df = pd.read_csv(csv_path)
        assert len(df) == 1, "Duplicate rows for same (model, ell)!"
        assert df.iloc[0]['accuracy'] == 97.2, "Old value not replaced"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
