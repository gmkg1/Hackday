"""
Code Generator Service — Journey D (AI Generates Cleaning Code).

This service handles ONLY the prompt construction and code extraction.
It builds dataset-aware prompts and processes Gemini responses into
valid Python cleaning code.

Separated from gemini_api.py so that:
    - gemini_api.py = pure API communication (reusable for any LLM call)
    - code_generator.py = cleaning-specific prompt logic + code extraction

Data Cleaning Strategy:
    The generated code addresses ALL of the following:
    1. Missing values — median (numeric), mode (categorical), drop (>50% missing)
    2. Duplicates — removed
    3. Outlier detection — IQR-based capping or Z-score filtering
    4. Class imbalance — SMOTE for classification if minority < 30% of majority
    5. Feature encoding — one-hot for low cardinality, label encoding for high
    6. Feature scaling — StandardScaler for numeric features
    7. High cardinality — columns with >50% unique values are dropped (except target)
    8. Data leakage — features perfectly correlated with target are flagged
"""
import json
import re
from typing import Dict, Any


from services.gemini_api import call_gemini, is_api_available


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

# ==============================================================================
#                      DATA CLEANING CODE PROMPT TEMPLATE
# ==============================================================================
# This prompt instructs Gemini to generate a data cleaning script (Journey D).
# It passes detailed profiling intelligence including missing value percentages,
# basic stats, correlation heatmaps, class imbalances, outlier heuristics,
# and data leakage flags.
#
# Key Features:
#   1. Input: Interpolated metadata variables (shape, cols, types, cardinality,
#             missing values, outliers, imbalance details, sample rows).
#   2. Output: Must return ONLY executable Python code defining a function
#      `def clean(df: pd.DataFrame) -> pd.DataFrame:`.
#   3. Preprocessing requirements enforced:
#      - Missing value imputations (median/mode/dropping).
#      - Exact duplicate removals.
#      - IQR-based outlier clipping.
#      - Imbalance treatment (SMOTE upsampling if <30%).
#      - Feature encoding (one-hot vs label).
#      - Scaling (StandardScaler).
#      - High cardinality dropping.
# ==============================================================================
CLEANING_CODE_PROMPT = """\
You are an expert data scientist. Given the dataset metadata below, write a \
comprehensive Python preprocessing function that cleans the dataframe for a \
{problem_type} task predicting the target column "{target_column}".

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

### Class Imbalance Analysis
{imbalance_section}

### Outlier Summary
{outlier_section}

### Sample Rows (from head, middle, and tail of dataset)
{sample_rows_section}

## Cleaning Requirements (MUST address ALL)

### 1. Missing Values
- Numeric columns with < 30% missing → impute with MEDIAN
- Numeric columns with 30-50% missing → impute with median, add a binary indicator column `<col>_was_missing`
- Numeric columns with > 50% missing → DROP the column
- Categorical columns → impute with MODE, or "Unknown" if mode is ambiguous
- NEVER drop rows unless > 80% of the row is NaN

### 2. Duplicate Rows
- Remove exact duplicate rows using df.drop_duplicates()

### 3. Outlier Handling
- For numeric features, apply IQR-based capping:
  - Lower bound = Q1 - 1.5 * IQR
  - Upper bound = Q3 + 1.5 * IQR
  - Clip values to [lower, upper] (do NOT remove rows)
- Skip outlier handling for the target column

### 4. Class Imbalance (Classification ONLY)
{imbalance_instructions}

### 5. Feature Encoding
- Categorical columns with ≤ 10 unique values → pd.get_dummies() (one-hot, drop_first=True)
- Categorical columns with > 10 unique values → LabelEncoder from sklearn.preprocessing
- Boolean columns → convert to int (0/1)

### 6. Feature Scaling
- Apply StandardScaler to ALL numeric feature columns (NOT the target column)
- Import StandardScaler from sklearn.preprocessing

### 7. High Cardinality
- Drop columns where nunique > 50% of total rows (these are likely IDs) — UNLESS it's the target column

### 8. Data Leakage Check
- If any feature has correlation > 0.95 with the target, add a comment warning about possible leakage but do NOT drop it automatically

## Function Signature

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # ... cleaning logic ...
    return df
```

## Rules
1. Import pandas as pd, numpy as np, and any sklearn utilities at the TOP of the script.
2. Do NOT drop the target column "{target_column}".
3. Do NOT perform train/test split.
4. Add brief inline comments explaining each cleaning step.
5. Return the cleaned DataFrame.

## Output
Return ONLY valid Python code. No markdown fences, no explanations outside the code.
"""


# ==============================================================================
#                      CODE CORRECTION PROMPT TEMPLATE
# ==============================================================================
# Used for the Self-Correction loop in the code generator service. If the LLM's
# generated preprocessing code throws a syntax or execution error inside the
# sandbox, this prompt feeds the traceback and erroneous code back to the LLM.
# It demands a corrected, fully executable version of the function without any
# textual explanations.
# ==============================================================================
CORRECTION_PROMPT = """\
The following Python preprocessing code was generated for a {problem_type} \
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
- Shape: {num_rows} rows × {num_cols} columns
- Columns: {columns}
- Column types: {column_types}
- Target column: {target_column}

## Instructions
1. Fix the code so it runs without errors.
2. Keep the same function signature: def clean(df: pd.DataFrame) -> pd.DataFrame
3. Import pandas as pd, numpy as np, and any needed sklearn utilities at the top.
4. Return ONLY valid Python code. No markdown fences.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cleaning_code(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a comprehensive cleaning prompt from dataset metadata and call
    Gemini to generate a clean(df) function.

    The prompt includes:
        - Missing value analysis with percentages
        - Basic statistics (mean, std, min, max)
        - Correlation matrix (top correlations)
        - Class imbalance detection with SMOTE instructions
        - Outlier summary (IQR-based)
        - Diverse sample rows (head + middle + tail)

    Args:
        metadata: Rich dataset metadata from profiler.build_llm_metadata()

    Returns:
        dict with:
            - "code": the generated Python code string
            - "explanation": what the code does
            - "error": error message if generation failed, else None
    """
    if not is_api_available():
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": "Gemini API key not configured. Using fallback static cleaner.",
            "error": "GEMINI_API_KEY not set",
        }

    prompt = _build_cleaning_prompt(metadata)

    response = call_gemini(prompt, temperature=0.2)

    if response["error"]:
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": f"Gemini call failed: {response['error']}. Using fallback.",
            "error": response["error"],
        }

    code = _extract_python_code(response["text"])

    if "def clean(" not in code:
        return {
            "code": code,
            "explanation": "Warning: generated code may not contain clean(df) function.",
            "error": "Missing clean() function definition",
        }

    return {
        "code": code,
        "explanation": "AI-generated cleaning code covering missing values, outliers, "
                       "encoding, scaling, and class imbalance handling.",
        "error": None,
    }


def regenerate_cleaning_code(
    metadata: Dict[str, Any],
    previous_code: str,
    error_traceback: str,
) -> Dict[str, Any]:
    """
    Self-correction: send failed code + traceback to Gemini for a fix.

    Args:
        metadata: Dataset metadata for context
        previous_code: The code that failed
        error_traceback: The Python traceback from execution

    Returns:
        dict with code, explanation, error
    """
    if not is_api_available():
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": "No API key for retry. Fallback.",
            "error": "GEMINI_API_KEY not set",
        }

    prompt = CORRECTION_PROMPT.format(
        problem_type=metadata["problem_type"],
        previous_code=previous_code,
        error_traceback=error_traceback,
        num_rows=metadata["num_rows"],
        num_cols=metadata["num_cols"],
        columns=metadata["columns"],
        column_types=json.dumps(metadata["column_types"]),
        target_column=metadata["target_column"],
    )

    response = call_gemini(prompt, temperature=0.1)

    if response["error"]:
        return {
            "code": _fallback_cleaning_code(metadata),
            "explanation": f"Retry failed: {response['error']}. Using fallback.",
            "error": response["error"],
        }

    code = _extract_python_code(response["text"])
    return {
        "code": code,
        "explanation": "Self-corrected code after execution failure.",
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cleaning_prompt(metadata: Dict[str, Any]) -> str:
    """
    Assemble the comprehensive cleaning prompt from dataset metadata.

    Enriches the prompt with:
        - Imbalance analysis (computes imbalance ratio from class distribution)
        - Outlier summary (from basic stats: IQR-based detection)
        - Specific SMOTE instructions for imbalanced classification datasets
    """

    # ── Missing values section ────────────────────────────────────────────
    if metadata["missing_values"]:
        missing_lines = []
        for col, count in metadata["missing_values"].items():
            pct = metadata["missing_percentages"].get(col, 0)
            severity = "LOW" if pct < 30 else ("MEDIUM" if pct < 50 else "HIGH")
            missing_lines.append(
                f"  - {col}: {count} missing ({pct}%) — Severity: {severity}"
            )
        missing_section = "\n".join(missing_lines)
    else:
        missing_section = "  No missing values detected."

    # ── Basic stats section ───────────────────────────────────────────────
    if metadata["basic_stats"]:
        stats_lines = []
        for col, stats in metadata["basic_stats"].items():
            stats_lines.append(
                f"  - {col}: mean={stats.get('mean', 'N/A')}, "
                f"std={stats.get('std', 'N/A')}, "
                f"min={stats.get('min', 'N/A')}, "
                f"25%={stats.get('25%', 'N/A')}, "
                f"50%={stats.get('50%', 'N/A')}, "
                f"75%={stats.get('75%', 'N/A')}, "
                f"max={stats.get('max', 'N/A')}"
            )
        basic_stats_section = "\n".join(stats_lines)
    else:
        basic_stats_section = "  No numeric columns found."

    # ── Correlation section ───────────────────────────────────────────────
    if metadata["correlations"]:
        correlation_section = _format_top_correlations(metadata["correlations"])
    else:
        correlation_section = "  Not enough numeric columns for correlations."

    # ── Target section ────────────────────────────────────────────────────
    if metadata.get("class_distribution"):
        dist = metadata["class_distribution"]
        target_section = (
            f"  Classification target with {len(dist)} classes:\n"
            f"  {json.dumps(dist, indent=2)}"
        )
    elif metadata.get("target_stats"):
        stats = metadata["target_stats"]
        target_section = f"  Regression target stats: {json.dumps(stats)}"
    else:
        target_section = "  Target info unavailable."

    # ── Class imbalance section ───────────────────────────────────────────
    imbalance_section, imbalance_instructions = _analyze_imbalance(metadata)

    # ── Outlier section ───────────────────────────────────────────────────
    outlier_section = _analyze_outliers(metadata)

    # ── Sample rows section ───────────────────────────────────────────────
    sample_rows = metadata.get("sample_rows", [])
    if sample_rows:
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
        imbalance_section=imbalance_section,
        imbalance_instructions=imbalance_instructions,
        outlier_section=outlier_section,
        sample_rows_section=sample_rows_section,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _analyze_imbalance(metadata: Dict[str, Any]) -> tuple:
    """
    Analyze class distribution and determine if SMOTE is needed.

    Returns:
        (imbalance_section_text, imbalance_instructions_text)
    """
    if metadata["problem_type"] != "classification":
        return (
            "  Not applicable (regression task).",
            "- Not applicable for regression. Skip this section.",
        )

    dist = metadata.get("class_distribution", {})
    if not dist:
        return (
            "  Class distribution unavailable.",
            "- Could not determine class balance. Skip SMOTE.",
        )

    counts = list(dist.values())
    majority = max(counts)
    minority = min(counts)
    total = sum(counts)
    imbalance_ratio = round(minority / majority, 3) if majority > 0 else 1.0

    lines = [f"  - Total samples: {total}"]
    for cls, cnt in dist.items():
        pct = round(cnt / total * 100, 1)
        lines.append(f"  - Class '{cls}': {cnt} samples ({pct}%)")
    lines.append(f"  - Imbalance ratio (minority/majority): {imbalance_ratio}")

    if imbalance_ratio < 0.3:
        lines.append("  - ⚠️ SIGNIFICANT IMBALANCE DETECTED — SMOTE recommended")
        instructions = """\
- SIGNIFICANT class imbalance detected (minority < 30% of majority).
- Apply SMOTE (Synthetic Minority Oversampling Technique):
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42)
  X = df.drop(columns=[target_column])
  y = df[target_column]
  X_resampled, y_resampled = smote.fit_resample(X, y)
  df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                   pd.DataFrame(y_resampled, columns=[target_column])], axis=1)
  ```
- SMOTE must be applied AFTER all encoding and scaling steps.
- Import: from imblearn.over_sampling import SMOTE
- If imblearn is not available, use random oversampling as fallback:
  ```python
  # Fallback: random oversampling
  from sklearn.utils import resample
  minority_class = df[df[target_col] == minority_label]
  majority_class = df[df[target_col] == majority_label]
  minority_upsampled = resample(minority_class, replace=True, 
                                 n_samples=len(majority_class), random_state=42)
  df = pd.concat([majority_class, minority_upsampled])
  ```"""
    elif imbalance_ratio < 0.5:
        lines.append("  - ⚡ Moderate imbalance — consider class_weight='balanced' in model")
        instructions = """\
- Moderate class imbalance detected.
- Do NOT apply SMOTE (imbalance is manageable).
- Instead, add a comment recommending class_weight='balanced' during model training.
- No resampling needed in the cleaning step."""
    else:
        lines.append("  - ✅ Classes are reasonably balanced")
        instructions = """\
- Classes are reasonably balanced. No resampling needed.
- Skip this section."""

    return "\n".join(lines), instructions


def _analyze_outliers(metadata: Dict[str, Any]) -> str:
    """
    Detect potential outliers from basic statistics using IQR heuristic.

    Checks if (max - Q3) > 3 * IQR or (Q1 - min) > 3 * IQR
    to flag columns with extreme outliers.
    """
    stats = metadata.get("basic_stats", {})
    if not stats:
        return "  No numeric columns to analyze for outliers."

    outlier_lines = []
    for col, s in stats.items():
        q1 = s.get("25%")
        q3 = s.get("75%")
        col_min = s.get("min")
        col_max = s.get("max")

        if None in (q1, q3, col_min, col_max):
            continue

        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_extreme = (q1 - col_min) > 3 * iqr
        upper_extreme = (col_max - q3) > 3 * iqr

        if lower_extreme or upper_extreme:
            direction = []
            if lower_extreme:
                direction.append(f"low (min={col_min} vs Q1={q1})")
            if upper_extreme:
                direction.append(f"high (max={col_max} vs Q3={q3})")
            outlier_lines.append(
                f"  - {col}: Extreme outliers detected — {', '.join(direction)}, IQR={round(iqr, 2)}"
            )

    if not outlier_lines:
        return "  No extreme outliers detected across numeric columns."

    return "\n".join(outlier_lines)


def _format_top_correlations(correlations: Dict, top_n: int = 5) -> str:
    """Extract and format the top N strongest correlations (excluding self)."""
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
    for col_a, col_b, abs_val, value in pairs[:top_n]:
        flag = " ⚠️ POSSIBLE LEAKAGE" if abs_val > 0.95 else ""
        lines.append(f"  - {col_a} ↔ {col_b}: {value}{flag}")
    return "\n".join(lines)


def _extract_python_code(text: str) -> str:
    """Strip markdown code fences if the LLM wrapped the code."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

def _fallback_cleaning_code(metadata: Dict[str, Any]) -> str:
    """
    Static fallback when Gemini is unavailable. Covers:
        - Duplicates, missing values, encoding, scaling, outlier capping
        - SMOTE via random oversampling for imbalanced classification
    """
    target = metadata.get("target_column", "target")
    problem_type = metadata.get("problem_type", "classification")

    smote_block = ""
    if problem_type == "classification":
        dist = metadata.get("class_distribution", {})
        if dist:
            counts = list(dist.values())
            ratio = min(counts) / max(counts) if max(counts) > 0 else 1.0
            if ratio < 0.3:
                smote_block = f'''
    # ── Class Imbalance: Random oversampling (SMOTE fallback) ─────────
    target_col = "{target}"
    from sklearn.utils import resample
    class_counts = df[target_col].value_counts()
    majority_label = class_counts.idxmax()
    minority_label = class_counts.idxmin()
    majority_df = df[df[target_col] == majority_label]
    minority_df = df[df[target_col] == minority_label]
    minority_upsampled = resample(
        minority_df, replace=True,
        n_samples=len(majority_df), random_state=42
    )
    df = pd.concat([majority_df, minority_upsampled]).reset_index(drop=True)
'''

    return f'''\
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive fallback cleaning function."""
    target_col = "{target}"

    # ── 1. Drop duplicate rows ────────────────────────────────────────
    df = df.drop_duplicates()

    # ── 2. Drop high-cardinality ID-like columns ─────────────────────
    for col in df.columns:
        if col != target_col and df[col].nunique() > len(df) * 0.5:
            df = df.drop(columns=[col])

    # ── 3. Handle missing values ─────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        if col == target_col:
            continue
        pct_missing = df[col].isnull().sum() / len(df) * 100
        if pct_missing > 50:
            df = df.drop(columns=[col])
        elif pct_missing > 0:
            if pct_missing >= 30:
                df[col + "_was_missing"] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if col == target_col:
            continue
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # ── 4. Outlier capping (IQR-based) ───────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == target_col:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)

    # ── 5. Encode categorical columns ────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in cat_cols if c != target_col and df[c].nunique() <= 10]
    high_card = [c for c in cat_cols if c != target_col and df[c].nunique() > 10]

    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=True)

    for col in high_card:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ── 6. Scale numeric features ────────────────────────────────────
    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != target_col]
    if numeric_features:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
{smote_block}
    return df
'''
