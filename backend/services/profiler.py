"""
Dataset Profiler Service — Journey B (Profiling) + Journey C (Problem Type Detection).

Pure business logic. No HTTP, no Flask imports.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


# ── Journey B: Dataset Profiling ──────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive profile of the dataset.

    Returns a dict containing:
        - num_rows, num_cols
        - missing_values (column → count)
        - missing_percentages (column → pct)
        - column_types (column → "numeric" / "categorical" / "datetime" / "boolean")
        - duplicate_rows count
        - basic_stats (describe() for numerics)
        - correlations (numeric correlation matrix as nested dict)
        - sample_rows (diverse sample from head, middle, tail)
        - columns list
    """
    num_rows, num_cols = df.shape

    # ── Missing values ────────────────────────────────────────────────────
    missing_counts = df.isnull().sum()
    missing_values = {
        col: int(count)
        for col, count in missing_counts.items()
        if count > 0
    }
    missing_percentages = {
        col: round(count / num_rows * 100, 2)
        for col, count in missing_counts.items()
        if count > 0
    }

    # ── Column type classification ────────────────────────────────────────
    column_types = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            column_types[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(dtype):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"

    # ── Duplicate rows ────────────────────────────────────────────────────
    duplicate_rows = int(df.duplicated().sum())

    # ── Basic statistics (numeric columns) ────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    basic_stats = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().round(4)
        basic_stats = desc.to_dict()

    # ── Correlation matrix (numeric columns) ──────────────────────────────
    correlations = {}
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr().round(4)
        correlations = corr.to_dict()

    # ── Diverse sample rows (head + middle + tail) ────────────────────────
    sample_rows = _get_diverse_sample(df, n=10)

    # ── Cardinality info ──────────────────────────────────────────────────
    cardinality = {
        col: int(df[col].nunique())
        for col in df.columns
    }

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": list(df.columns),
        "column_types": column_types,
        "missing_values": missing_values,
        "missing_percentages": missing_percentages,
        "duplicate_rows": duplicate_rows,
        "basic_stats": basic_stats,
        "correlations": correlations,
        "cardinality": cardinality,
        "sample_rows": sample_rows,
    }


def _get_diverse_sample(df: pd.DataFrame, n: int = 10) -> list:
    """
    Pull rows from head, middle, and tail of the dataset so the LLM
    sees diverse data — not just the first few rows which may be atypical.

    Strategy:
        - 3 rows from head
        - 4 rows from random middle positions
        - 3 rows from tail
    Capped at total n rows, deduplicated by index.
    """
    total = len(df)
    if total <= n:
        return df.to_dict(orient="records")

    head_n = min(3, n // 3)
    tail_n = min(3, n // 3)
    mid_n = n - head_n - tail_n

    indices = set()

    # Head indices
    indices.update(range(0, head_n))

    # Tail indices
    indices.update(range(total - tail_n, total))

    # Middle indices — evenly spaced through the middle 60% of the dataset
    mid_start = int(total * 0.2)
    mid_end = int(total * 0.8)
    if mid_end > mid_start:
        mid_positions = np.linspace(mid_start, mid_end - 1, mid_n, dtype=int)
        indices.update(mid_positions.tolist())

    sorted_indices = sorted(indices)[:n]
    sample_df = df.iloc[sorted_indices].copy()

    # Convert to records, handling NaN → None for JSON serialization
    records = sample_df.where(sample_df.notna(), None).to_dict(orient="records")
    return records


# ── Journey C: Problem Type Detection ─────────────────────────────────────────

def detect_problem_type(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Auto-detect the target column and whether the problem is
    classification or regression.

    Heuristic:
        1. If target_column is given, use it.
        2. Otherwise pick the last column.
        3. If unique values ≤ 10  OR  dtype is object/category → classification.
        4. Else → regression.

    Returns dict with:
        - target_column
        - problem_type ("classification" | "regression")
        - num_classes (for classification)
        - class_distribution (for classification)
        - target_stats (for regression)
    """
    # ── Resolve target column ─────────────────────────────────────────────
    if target_column and target_column in df.columns:
        target = target_column
    else:
        target = df.columns[-1]

    target_series = df[target]
    n_unique = target_series.nunique()
    dtype = target_series.dtype

    # ── Classification vs Regression ──────────────────────────────────────
    is_classification = (
        n_unique <= 10
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_categorical_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )

    result: Dict[str, Any] = {
        "target_column": target,
    }

    if is_classification:
        result["problem_type"] = "classification"
        result["num_classes"] = int(n_unique)
        distribution = target_series.value_counts().to_dict()
        # Convert numpy types to plain Python for JSON serialization
        result["class_distribution"] = {
            str(k): int(v) for k, v in distribution.items()
        }
    else:
        result["problem_type"] = "regression"
        stats = target_series.describe().to_dict()
        result["target_stats"] = {
            k: float(v) for k, v in stats.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }

    return result


# ── Build Metadata Payload for LLM ───────────────────────────────────────────

def build_llm_metadata(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    problem_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble a rich metadata payload that the LLM will use to write
    smart, dataset-specific cleaning code.

    Includes:
        - shape, columns, types, cardinality
        - missing value counts AND percentages
        - basic stats (mean, std, min, max, quartiles)
        - correlations between numeric features
        - class distribution / target stats
        - diverse sample rows (head + middle + tail)
    """
    return {
        "num_rows": profile["num_rows"],
        "num_cols": profile["num_cols"],
        "columns": profile["columns"],
        "column_types": profile["column_types"],
        "missing_values": profile["missing_values"],
        "missing_percentages": profile["missing_percentages"],
        "duplicate_rows": profile["duplicate_rows"],
        "cardinality": profile["cardinality"],
        "basic_stats": profile["basic_stats"],
        "correlations": profile["correlations"],
        "target_column": problem_info["target_column"],
        "problem_type": problem_info["problem_type"],
        "class_distribution": problem_info.get("class_distribution"),
        "target_stats": problem_info.get("target_stats"),
        "sample_rows": profile["sample_rows"],
    }
