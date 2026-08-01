"""
End-to-end test for the AutoML Scientist pipeline (Journey A → D).

Creates a realistic sample dataset with:
  - Missing values
  - Class imbalance
  - Outliers
  - Mixed column types
  - Duplicates

Then runs: Upload → Profile → Problem Detection → Code Generation → Execution
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# ── Create a realistic test dataset ──────────────────────────────────────────

def create_test_dataset():
    """
    Build a Titanic-like dataset with known issues:
      - Missing values in Age (15%), Cabin (78%), Embarked (1%)
      - Class imbalance: ~63% did NOT survive, ~37% survived
      - Outliers in Fare
      - High cardinality in PassengerId, Name
      - Duplicates
    """
    np.random.seed(42)
    n = 200

    data = {
        "PassengerId": list(range(1, n + 1)),
        "Pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.30, 0.46]),
        "Name": [f"Passenger_{i}" for i in range(n)],
        "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
        "Age": np.random.normal(30, 14, n).round(1),
        "SibSp": np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        "Parch": np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
        "Fare": np.random.exponential(35, n).round(2),
        "Cabin": [f"C{np.random.randint(1,150)}" if np.random.random() > 0.78 else None for _ in range(n)],
        "Embarked": np.random.choice(["S", "C", "Q", None], n, p=[0.70, 0.19, 0.09, 0.02]),
        "Survived": np.random.choice([0, 1], n, p=[0.63, 0.37]),
    }

    df = pd.DataFrame(data)

    # Inject missing Age values (~15%)
    missing_age_idx = np.random.choice(n, size=int(n * 0.15), replace=False)
    df.loc[missing_age_idx, "Age"] = np.nan

    # Inject outliers in Fare
    df.loc[5, "Fare"] = 512.33
    df.loc[10, "Fare"] = 480.00
    df.loc[15, "Fare"] = 600.50

    # Inject duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return df


# ── Separator helper ─────────────────────────────────────────────────────────

def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ── Main Test ─────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Create dataset ────────────────────────────────────────────
    separator("STEP 1: Creating Test Dataset")
    df = create_test_dataset()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    # ── Step 2: Profile dataset (Journey B) ───────────────────────────────
    separator("STEP 2: Profiling Dataset (Journey B)")
    from services.profiler import profile_dataset

    profile = profile_dataset(df)

    print(f"Num rows: {profile['num_rows']}")
    print(f"Num cols: {profile['num_cols']}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")
    print(f"\nMissing values:")
    for col, count in profile["missing_values"].items():
        pct = profile["missing_percentages"][col]
        print(f"  {col}: {count} ({pct}%)")
    print(f"\nColumn types:")
    for col, ctype in profile["column_types"].items():
        print(f"  {col}: {ctype}")
    print(f"\nCardinality:")
    for col, card in profile["cardinality"].items():
        print(f"  {col}: {card} unique")
    print(f"\nSample rows ({len(profile['sample_rows'])} rows from head+middle+tail):")
    for i, row in enumerate(profile["sample_rows"][:3]):
        print(f"  Row {i}: {row}")
    print(f"  ... ({len(profile['sample_rows'])} total)")

    # ── Step 3: Detect problem type (Journey C) ──────────────────────────
    separator("STEP 3: Problem Type Detection (Journey C)")
    from services.profiler import detect_problem_type

    problem_info = detect_problem_type(df, target_column="Survived")

    print(f"Target column: {problem_info['target_column']}")
    print(f"Problem type: {problem_info['problem_type']}")
    if problem_info["problem_type"] == "classification":
        print(f"Num classes: {problem_info['num_classes']}")
        print(f"Class distribution: {problem_info['class_distribution']}")
        dist = problem_info["class_distribution"]
        counts = list(dist.values())
        ratio = min(counts) / max(counts) if max(counts) > 0 else 1
        print(f"Imbalance ratio: {ratio:.3f}")
        if ratio < 0.3:
            print("[!] SMOTE would be applied (ratio < 0.3)")
        elif ratio < 0.5:
            print("[~] Moderate imbalance -- class_weight='balanced' recommended")
        else:
            print("[OK] Reasonably balanced")

    # ── Step 4: Build LLM metadata ────────────────────────────────────────
    separator("STEP 4: Building LLM Metadata")
    from services.profiler import build_llm_metadata

    metadata = build_llm_metadata(df, profile, problem_info)
    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"Sample rows included: {len(metadata['sample_rows'])}")
    print(f"Stats for {len(metadata['basic_stats'])} numeric columns")
    print(f"Correlations for {len(metadata['correlations'])} column pairs")

    # ── Step 5: Generate cleaning code (Journey D) ────────────────────────
    separator("STEP 5: AI Code Generation (Journey D)")
    from services.code_generator import generate_cleaning_code

    print("Calling Gemini to generate clean(df) code...")
    print("(If GEMINI_API_KEY is not set, fallback code will be used)\n")

    result = generate_cleaning_code(metadata)

    if result["error"]:
        print(f"[!] Note: {result['error']}")
    print(f"Explanation: {result['explanation']}")
    print(f"\n{'-'*50}")
    print("GENERATED CODE:")
    print(f"{'-'*50}")
    # Force ASCII encode/decode for Windows cmd compatibility
    print(result["code"].encode("ascii", "ignore").decode("ascii"))
    print(f"{'-'*50}")

    # ── Step 6: Execute the generated code ────────────────────────────────
    separator("STEP 6: Executing Generated Code")
    from services.executor import run_generated_code

    print(f"Running clean(df) on {len(df)} rows...")
    cleaned_df, error = run_generated_code(result["code"], df)

    if error:
        print(f"\n[FAIL] EXECUTION FAILED:")
        print(error)

        # Try self-correction
        print("\n[RETRY] Attempting self-correction...")
        from services.code_generator import regenerate_cleaning_code
        retry_result = regenerate_cleaning_code(metadata, result["code"], error)
        print(f"\nRetry explanation: {retry_result['explanation']}")

        cleaned_df, error2 = run_generated_code(retry_result["code"], df)
        if error2:
            print(f"\n[FAIL] RETRY ALSO FAILED:")
            print(error2)
        else:
            print(f"\n[OK] Self-correction SUCCEEDED!")
            cleaned_df_info(cleaned_df, df)
    else:
        print(f"\n[OK] EXECUTION SUCCEEDED!")
        cleaned_df_info(cleaned_df, df)


def cleaned_df_info(cleaned_df, original_df):
    """Print comparison between original and cleaned dataframes."""
    print(f"\n{'-'*50}")
    print("BEFORE vs AFTER COMPARISON:")
    print(f"{'-'*50}")
    print(f"  Rows:    {len(original_df)} -> {len(cleaned_df)} ({'+'if len(cleaned_df)>=len(original_df) else ''}{len(cleaned_df)-len(original_df)})")
    print(f"  Columns: {len(original_df.columns)} -> {len(cleaned_df.columns)} ({'+'if len(cleaned_df.columns)>=len(original_df.columns) else ''}{len(cleaned_df.columns)-len(original_df.columns)})")
    print(f"  Missing: {original_df.isnull().sum().sum()} -> {cleaned_df.isnull().sum().sum()}")
    print(f"  Dtypes:  {dict(cleaned_df.dtypes.value_counts())}")
    print(f"\nCleaned columns: {list(cleaned_df.columns)}")
    print(f"\nFirst 3 rows of cleaned data:")
    print(cleaned_df.head(3).to_string())


if __name__ == "__main__":
    main()
