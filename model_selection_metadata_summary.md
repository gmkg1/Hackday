# Dataset Metadata and Prompt Configurations Sent to AI for ML Model Selection

This document summarizes the exact metadata schemas, empirical performance metrics, prompts, and rules sent to the AI models during the autonomous machine learning model selection process. All details are extracted directly from the python scripts in the codebase.

---

## 1. Architecture Overview of Model Selection

Model selection occurs in **Journey F (Autonomous Research Loop)**. It consists of a dual-layered AI system:

1. **AutoML Scientist (Groq/Llama-3)**: The primary logic generator. It receives the dataset metadata, statistical baseline scores, and historical trial history (compiled in a **Dataset Intelligence JSON**) and selects the optimal ML model family and hyperparameters. It then generates the corresponding Python training script.
2. **Collaborative Reviewer (Gemini)**: A review layer that executes after each sandbox run. It evaluates detailed model performance metrics (like classification reports or feature importances) and updates the instructions for the AutoML Scientist in the next iteration.

---

## 2. The Dataset Intelligence JSON (AutoML Scientist Input)

The primary dataset metadata payload is compiled by the function `compile_dataset_intelligence` in [ml_engine.py](file:///c:/Programs/Hackday-1/backend/services/ml_engine.py#L89) and structured in [orchestrate_handler.py](file:///c:/Programs/Hackday-1/backend/handlers/orchestrate_handler.py#L145).

The compiled JSON payload consists of the following key groups:

### 2.1. `session`
Tracks the context of the active optimization session.
* **`session_id`** *(string)*: Unique identifier of the session.
* **`dataset_name`** *(string)*: The name of the uploaded dataset file (extracted from `ds_state.filename`).
* **`experiment_iteration`** *(integer)*: The current trial iteration (ranges from 1 up to `max_iterations`).

### 2.2. `dataset`
Describes the physical shape and basic parameters of the cleaned dataset.
* **`problem_type`** *(string)*: Heuristically detected target type: `"classification"` or `"regression"` (computed by `detect_problem_type` in [profiler.py](file:///c:/Programs/Hackday-1/backend/services/profiler.py#L142)).
* **`target_column`** *(string)*: Name of the target variable to be predicted.
* **`rows`** *(integer)*: Number of rows in the cleaned DataFrame.
* **`columns`** *(integer)*: Number of columns in the cleaned DataFrame.
* **`memory_usage_mb`** *(float)*: Deep memory footprint of the cleaned DataFrame in MB.
* **`duplicate_rows`** *(integer)*: Quantity of exact duplicate rows remaining in the cleaned DataFrame.

### 2.3. `missing_values`
Contains the pre-cleaning profile of column missing counts for context.
* **`missing_values`** *(dict)*: Key-value mapping of `{ column_name: missing_count }` for columns with missing values.

### 2.4. `basic_stats`
Provides descriptive statistics of the numeric features (pre-cleaning profile) to help the AI detect distributions and ranges.
* **`basic_stats`** *(dict)*: Nested dictionary containing pandas `describe()` statistics (count, mean, std, min, 25%, 50%, 75%, max) for each numeric column.

### 2.5. `target_analysis`
Describes the specific target distribution depending on the problem type.
* **`class_distribution`** *(dict)*: Frequencies of each target class for classification problems (e.g. `{"0": 500, "1": 300}`).
* **`target_stats`** *(dict)*: Summary statistics of the target series for regression problems (mean, std, min, percentiles, max).

### 2.6. `correlation`
Summarizes feature collinearity.
* **`average`** *(float)*: The mean of the absolute values in the upper triangle of the Pearson correlation matrix (excluding diagonal) across all numeric features in the cleaned DataFrame.

### 2.7. `baseline_models`
A list of baseline model runs evaluated locally via 5-fold cross-validation (implemented in `evaluate_baselines` in [ml_engine.py](file:///c:/Programs/Hackday-1/backend/services/ml_engine.py#L20)). This anchors the AI to baseline performance scores.
* **Classification Baselines**: "Logistic Regression" and "Decision Tree" (using `accuracy` and `f1_macro` metrics).
* **Regression Baselines**: "Ridge Regression" and "Decision Tree Regressor" (using `r2` and `neg_mean_squared_error` metrics).
* **Baseline Fields**:
  * **`model`** *(string)*: The name of the baseline model.
  * **`cv_score`** *(float)*: Mean cross-validation score (accuracy or $R^2$).
  * **`training_score`** *(float)*: Score on the training split.
  * **`validation_score`** *(float)*: Same as `cv_score`.
  * **`training_time`** *(float)*: Time taken to train the baseline model (in seconds).
  * **`prediction_time`** *(float)*: Hardcoded reference inference speed (`0.01`).
  * **`overfit_gap`** *(float)*: Difference between `training_score` and `validation_score`.

### 2.8. `experiment_history`
Tracks the history of model selection trials inside the loop, preventing the AI from repeating failed or suboptimal settings.
* **`iteration`** *(integer)*: Trial iteration count.
* **`model`** *(string)*: Name of the model chosen in that iteration.
* **`score`** *(float)*: Resulting validation score.
  * **`diagnosis`** *(string)*: Diagnosis of performance / issues recorded during that run.
  * **`result`** *(string)*: The execution status (e.g., `"Success"` or `"Failed: <error>"`).

---

## 3. Collaborative Review Metrics (Gemini Reviewer Input)

After executing the Python code proposed by the AutoML Scientist, the system computes advanced metrics and submits them to the Gemini Collaboration Layer. This logic is housed in `compute_evaluation_metrics` and the reviewer block in [orchestrate_handler.py](file:///c:/Programs/Hackday-1/backend/handlers/orchestrate_handler.py#L415).

The metadata payload compiled for Gemini contains:

### 3.1. General Run Context
* **`iteration`** *(integer)*: Current loop iteration.
* **`model trained`** *(string)*: Name of the estimator trained.
* **`hyperparameters used`** *(json)*: The parameters supplied to the estimator constructor.
* **`problem_type`** *(string)*: `"classification"` or `"regression"`.
* **`target_column`** *(string)*: Target variable name.

### 3.2. Evaluation Metrics Block
* **`feature_importances`** *(dict)*: Mapping of column names to relative importance weights (sourced from `feature_importances_` or mean absolute coefficients `coef_`).
* **`summary`** *(string)*: Text summarizing key statistics (e.g. `"Accuracy: 0.8950, Macro F1: 0.8872"`).
* **For Classification Problems**:
  * **`accuracy`** *(float)*: Global accuracy score.
  * **`classification_report`** *(dict)*: Standard classification metrics (precision, recall, f1-score, support) for each class.
  * **`confusion_matrix`** *(list of lists)*: 2D array representation of correct/incorrect predictions.
* **For Regression Problems**:
  * **`r2_score`** *(float)*: Coefficient of determination.
  * **`mean_squared_error`** *(float)*: Mean Squared Error.
  * **`mean_absolute_error`** *(float)*: Mean Absolute Error.

### 3.3. Training Code Sourced
* **`training_code`** *(string)*: The full Python script that was executed.

---

## 4. Prompt Constraints and Selection Rules Enforced on AI

In addition to structural metadata, `MASTER_PROMPT` in [orchestrator.py](file:///c:/Programs/Hackday-1/backend/services/orchestrator.py#L37) contains strict behavioral constraints governing how the AI selects models:

1. **Evidence-based Selection**: The AI must explain *why* a model family was selected based on the metadata characteristics.
2. **Algorithm/Complexity Matching**:
   * For **simple or linear datasets**, regularized linear models (e.g., Ridge, Logistic Regression) must be prioritized.
   * For **complex or non-linear datasets**, tree-based ensembles (e.g., GradientBoostingClassifier, XGBoostClassifier, RandomForestClassifier, DecisionTreeClassifier) must be chosen.
3. **Selection Diversity**: The AI must not repeatedly output the same model family (e.g. RandomForest) across iterations.
4. **Historical Guidance**: The AI is prohibited from repeating an experiment that already failed unless a significant modification is introduced.
5. **Hyperparameter Restrictions**: The prompt explicitly warns against passing the `'class_weight'` parameter to Gradient Boosting (`GradientBoostingClassifier`/`GradientBoostingRegressor`) and XGBoost (`XGBClassifier`/`XGBRegressor`) models because scikit-learn and xgboost do not support it for these estimators.

---

## 5. Summary Table of Metadata Sources

| Metadata Category | Sourced From Function / Variable | Python Script Reference |
| :--- | :--- | :--- |
| **Row/Column Shape** | `df_clean.shape` | [ml_engine.py](file:///c:/Programs/Hackday-1/backend/services/ml_engine.py#L102) |
| **Memory Footprint** | `df_clean.memory_usage` | [ml_engine.py](file:///c:/Programs/Hackday-1/backend/services/ml_engine.py#L103) |
| **Collinearity Summary** | `df_clean.corr()` | [ml_engine.py](file:///c:/Programs/Hackday-1/backend/services/ml_engine.py#L107-L116) |
| **Deterministic Baselines** | `evaluate_baselines()` | [ml_engine.py](file:///c:/Programs/Hackday-1/backend/services/ml_engine.py#L20-L86) |
| **Missing Values Count** | `profile_dataset()` | [profiler.py](file:///c:/Programs/Hackday-1/backend/services/profiler.py#L30-L41) |
| **Descriptive Columns Stats** | `profile_dataset()` | [profiler.py](file:///c:/Programs/Hackday-1/backend/services/profiler.py#L59-L64) |
| **Target Column & Classes** | `detect_problem_type()` | [profiler.py](file:///c:/Programs/Hackday-1/backend/services/profiler.py#L142-L201) |
| **Session Trial History** | `session.experiments` | [orchestrate_handler.py](file:///c:/Programs/Hackday-1/backend/handlers/orchestrate_handler.py#L133-L142) |
| **Empirical Sandbox Metrics** | `compute_evaluation_metrics()` | [orchestrate_handler.py](file:///c:/Programs/Hackday-1/backend/handlers/orchestrate_handler.py#L415-L509) |
