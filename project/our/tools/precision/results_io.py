"""
Results I/O for the E0 precision study.

Merges per-model results into a shared CSV so that running one model's sweep
does not clobber another model's previously-saved results.
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd


def merge_results(csv_path, new_rows: List[Dict]) -> None:
    """
    Merge new result rows into an existing CSV, keyed by (model, ell).

    - If the CSV does not exist, it is created with new_rows.
    - Rows for models present in new_rows replace any existing rows with the
      same (model, ell) key.
    - Rows for other models are preserved.

    Args:
        csv_path: path to the results CSV.
        new_rows: list of dicts with keys 'model', 'ell', 'accuracy'.
    """
    csv_path = Path(csv_path)
    new_df = pd.DataFrame(new_rows)

    # ``ell`` contains both numeric widths and the ``float32`` reference row.
    # Pandas otherwise infers the old and new integer values with different
    # object dtypes, so duplicate keys survive the merge.  Normalize the key
    # before concatenating to make reruns idempotent.
    new_df['ell'] = new_df['ell'].astype(str)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, dtype={'ell': str})
        existing_df['ell'] = existing_df['ell'].astype(str)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Keep the last occurrence of each (model, ell) key (new_rows win)
        combined = combined.drop_duplicates(subset=['model', 'ell'], keep='last')
    else:
        combined = new_df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
