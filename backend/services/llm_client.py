"""
LLM Client Service — Journey D (AI Generates Cleaning Code).

Uses Google Gemini (google-genai SDK) to generate dataset-specific preprocessing code.
The prompt includes rich metadata (stats, correlations, diverse sample rows)
so the model writes feasible, data-aware cleaning logic.
"""
import json
import re
from typing import Dict, Any

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL


# ── Configure Gemini client ──────────────────────────────────────────────────

_client = None

def _get_client():
    """Lazy-init the Gemini client."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Prompt Template ───────────────────────────────────────────────────────────

CLEANING_CODE_PROMPT = """\
You are an expert data scientist. Given the dataset metadata below, write a Python \
preprocessing function that cleans the dataframe for a {problem_type} task \
predicting the target column "{target_column}".

## Dataset Metadata

- **Shape:** {num_rows} rows × {num_cols} columns
- **Columns:** {columns}
- **Column types:** {column_types}
- **Cardinality:** {cardinality}

### Missing Values
{missing_section}

### Basic Statistics (numeric columns)
{basic_stats_section}

### Correlation Matrix (top correlations)
{correlation_section}

### Target Info
{target_section}

### Sample Rows (from head, middle, and tail of dataset)
{sample_rows_section}

## Requirements

1. Write ONLY a single Python function with this exact signature:
   ```python
   def clean(df: pd.DataFrame) -> pd.DataFrame:
   ```
2. Import pandas as pd and numpy as np at the top of the script.
3. Handle missing values appropriately:
   - For numeric columns with < 30% missing: impute with median.
   - For numeric columns with >= 30% missing: consider dropping the column.
   - For categorical columns: impute with mode or "Unknown".
4. Encode categorical columns using pd.get_dummies() or label encoding.
5. Remove duplicate rows if any exist.
6. Drop columns with extremely high cardinality (> 50% of rows) unless they are the target.
7. Do NOT drop the target column "{target_column}".
8. Do NOT perform train/test split — just return the cleaned DataFrame.
9. Return the cleaned DataFrame.
10. Add brief inline comments explaining each cleaning step.

## Output

Return ONLY valid Python code. No markdown fences, no explanations outside the code.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_cleaning_code(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Gemini to generate a clean(df) preprocessing function.

    Args:
        metadata: Rich dataset metadata from profiler.build_llm_metadata()

    Returns:
        dict with:
            - "code": the generated Python code string
            - "explanation": brief explanation of what the code does
            - "error": error message if generation failed, else None
    """
    if not GEMINI_API_KEY:
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": "Gemini API key not configured. Using fallback static cleaner.",
            "error": "GEMINI_API_KEY not set",
        }

    prompt = _build_prompt(metadata)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,  # Low temperature for deterministic code
            ),
        )
        raw_text = response.text

        # Extract pure Python code (strip markdown fences if present)
        code = _extract_python_code(raw_text)

        # Validate it defines clean(df)
        if "def clean(" not in code:
            return {
                "code": code,
                "explanation": "Warning: generated code may not contain clean(df) function.",
                "error": "Missing clean() function definition",
            }

        return {
            "code": code,
            "explanation": "AI-generated cleaning code based on dataset profile.",
            "error": None,
        }

    except Exception as e:
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": f"Gemini call failed: {str(e)}. Using fallback cleaner.",
            "error": str(e),
        }


def regenerate_cleaning_code(
    metadata: Dict[str, Any],
    previous_code: str,
    error_traceback: str,
) -> Dict[str, Any]:
    """
    Self-correction: resend the failed code + traceback to Gemini
    and ask it to fix the code.

    Used by the retry loop (Journey D self-correction, max 2 retries).
    """
    if not GEMINI_API_KEY:
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": "No API key for retry. Fallback.",
            "error": "GEMINI_API_KEY not set",
        }

    correction_prompt = f"""\
The following Python preprocessing code was generated for a {metadata['problem_type']} \
dataset but FAILED during execution.

## Failed Code
```python
{previous_code}
```

## Error Traceback
```
{error_traceback}
```

## Dataset Metadata (for reference)
- Shape: {metadata['num_rows']} rows × {metadata['num_cols']} columns
- Columns: {metadata['columns']}
- Column types: {metadata['column_types']}
- Target column: {metadata['target_column']}

## Instructions
1. Fix the code so it runs without errors.
2. Keep the same function signature: def clean(df: pd.DataFrame) -> pd.DataFrame
3. Import pandas as pd and numpy as np at the top.
4. Return ONLY valid Python code. No markdown fences.
"""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=correction_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Even lower for corrections
            ),
        )
        code = _extract_python_code(response.text)

        return {
            "code": code,
            "explanation": "Self-corrected code after execution failure.",
            "error": None,
        }
    except Exception as e:
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": f"Retry failed: {str(e)}. Using fallback.",
            "error": str(e),
        }


# ── Private Helpers ───────────────────────────────────────────────────────────

def _build_prompt(metadata: Dict[str, Any]) -> str:
    """Assemble the full prompt from metadata."""

    # Missing values section
    if metadata["missing_values"]:
        missing_lines = []
        for col, count in metadata["missing_values"].items():
            pct = metadata["missing_percentages"].get(col, 0)
            missing_lines.append(f"  - {col}: {count} missing ({pct}%)")
        missing_section = "\n".join(missing_lines)
    else:
        missing_section = "  No missing values detected."

    # Basic stats section
    if metadata["basic_stats"]:
        stats_lines = []
        for col, stats in metadata["basic_stats"].items():
            stats_lines.append(
                f"  - {col}: mean={stats.get('mean', 'N/A')}, "
                f"std={stats.get('std', 'N/A')}, "
                f"min={stats.get('min', 'N/A')}, "
                f"max={stats.get('max', 'N/A')}"
            )
        basic_stats_section = "\n".join(stats_lines)
    else:
        basic_stats_section = "  No numeric columns found."

    # Correlation section (show top 5 strongest correlations)
    if metadata["correlations"]:
        correlation_section = _format_top_correlations(metadata["correlations"])
    else:
        correlation_section = "  Not enough numeric columns for correlations."

    # Target section
    if metadata.get("class_distribution"):
        dist = metadata["class_distribution"]
        target_section = f"  Classification target with {len(dist)} classes: {json.dumps(dist)}"
    elif metadata.get("target_stats"):
        stats = metadata["target_stats"]
        target_section = f"  Regression target stats: {json.dumps(stats)}"
    else:
        target_section = "  Target info unavailable."

    # Sample rows section
    sample_rows = metadata.get("sample_rows", [])
    if sample_rows:
        # Show as a readable table-like format
        sample_rows_section = json.dumps(sample_rows[:10], indent=2, default=str)
    else:
        sample_rows_section = "  No sample rows available."

    return CLEANING_CODE_PROMPT.format(
        problem_type=metadata["problem_type"],
        target_column=metadata["target_column"],
        num_rows=metadata["num_rows"],
        num_cols=metadata["num_cols"],
        columns=metadata["columns"],
        column_types=json.dumps(metadata["column_types"], indent=2),
        cardinality=json.dumps(metadata["cardinality"], indent=2),
        missing_section=missing_section,
        basic_stats_section=basic_stats_section,
        correlation_section=correlation_section,
        target_section=target_section,
        sample_rows_section=sample_rows_section,
    )


def _format_top_correlations(correlations: Dict, top_n: int = 5) -> str:
    """Extract and format the top N strongest correlations (excluding self-correlation)."""
    pairs = []
    seen = set()
    for col_a, row in correlations.items():
        for col_b, value in row.items():
            if col_a == col_b:
                continue
            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pairs.append((col_a, col_b, abs(value), value))

    pairs.sort(key=lambda x: x[2], reverse=True)

    if not pairs:
        return "  No significant correlations."

    lines = []
    for col_a, col_b, _, value in pairs[:top_n]:
        lines.append(f"  - {col_a} ↔ {col_b}: {value}")
    return "\n".join(lines)


def _extract_python_code(text: str) -> str:
    """
    Strip markdown code fences if the LLM wrapped the code in ```python ... ```.
    Returns the raw Python code.
    """
    # Try to find ```python ... ``` blocks
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # Otherwise return as-is (already plain code)
    return text.strip()


def _fallback_cleaning_code(metadata: Dict[str, Any]) -> str:
    """
    Static fallback cleaning code when the LLM is unavailable.
    Handles the most common cleaning operations.
    """
    target = metadata.get("target_column", "target")
    return f'''\
import pandas as pd
import numpy as np

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback static cleaning function."""
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Separate target
    target_col = "{target}"

    # Handle missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != target_col and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Handle missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if col != target_col and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # Encode categorical columns (excluding target if it's categorical)
    encode_cols = [c for c in cat_cols if c != target_col]
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True)

    return df
'''
