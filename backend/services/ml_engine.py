"""
ML Engine Service — Journey E.

Responsibilities:
    - Train and evaluate models (baselines and LLM-proposed pipelines)
    - Handle train/test splitting and K-Fold cross-validation
    - Compile the "Dataset Intelligence JSON" for the LLM
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def evaluate_baselines(df: pd.DataFrame, target_column: str, problem_type: str) -> list:
    """
    Train and evaluate deterministic baselines to anchor the LLM.
    
    Returns a list of dicts with model performance metrics.
    """
    if target_column not in df.columns:
        return []

    # Drop any remaining NaNs just in case cleaning missed them
    df = df.dropna()
    if df.empty:
        return []

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert object columns to numeric (if not encoded properly by the cleaner)
    X = pd.get_dummies(X, drop_first=True)

    results = []

    if problem_type == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42))
        ]
        scoring = ['accuracy', 'f1_macro']
    else:
        models = [
            ("Ridge Regression", Ridge(random_state=42)),
            ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=5, random_state=42))
        ]
        scoring = ['r2', 'neg_mean_squared_error']

    for name, model in models:
        try:
            start_time = time.time()
            # 5-fold CV
            cv_results = cross_validate(
                model, X, y, cv=5, scoring=scoring, return_train_score=True, n_jobs=-1
            )
            training_time = time.time() - start_time

            if problem_type == "classification":
                val_score = cv_results['test_accuracy'].mean()
                train_score = cv_results['train_accuracy'].mean()
            else:
                val_score = cv_results['test_r2'].mean()
                train_score = cv_results['train_r2'].mean()

            results.append({
                "model": name,
                "cv_score": round(val_score, 4),
                "training_score": round(train_score, 4),
                "validation_score": round(val_score, 4),
                "training_time": round(training_time, 4),
                "prediction_time": 0.01,  # Approximation
                "overfit_gap": round(train_score - val_score, 4)
            })
        except Exception as e:
            results.append({
                "model": name,
                "error": str(e)
            })

    return results


def compile_dataset_intelligence(
    session_id: str,
    filename: str,
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    profile: Dict[str, Any],
    problem_info: Dict[str, Any],
    experiment_history: list
) -> Dict[str, Any]:
    """
    Compile the Dataset Intelligence JSON.
    This replaces sending raw dataframes to the LLM.
    """
    num_rows, num_cols = df_clean.shape
    memory_usage = round(df_clean.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    # Re-calculate some basic stats on the cleaned dataframe
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    correlation_matrix = df_clean[numeric_cols].corr() if len(numeric_cols) > 1 else pd.DataFrame()
    
    # Calculate average correlation
    avg_corr = 0.0
    if not correlation_matrix.empty:
        # Get upper triangle excluding diagonal
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        avg_corr = round(upper.abs().mean().mean(), 4)
        if pd.isna(avg_corr):
            avg_corr = 0.0

    baselines = evaluate_baselines(df_clean, problem_info['target_column'], problem_info['problem_type'])

    return {
        "session": {
            "session_id": session_id,
            "dataset_name": filename,
            "experiment_iteration": len(experiment_history) + 1
        },
        "dataset": {
            "problem_type": problem_info["problem_type"],
            "target_column": problem_info["target_column"],
            "rows": num_rows,
            "columns": num_cols,
            "memory_usage_mb": memory_usage,
            "duplicate_rows": int(df_clean.duplicated().sum())
        },
        "missing_values": profile.get("missing_values", {}), # Pre-cleaning profile for context
        "basic_stats": profile.get("basic_stats", {}),       # Pre-cleaning profile for context
        "target_analysis": {
            "class_distribution": problem_info.get("class_distribution", {}),
            "target_stats": problem_info.get("target_stats", {})
        },
        "correlation": {
            "average": avg_corr
        },
        "baseline_models": baselines,
        "experiment_history": experiment_history
    }
