# Backend Context — AutoML Scientist

## Project Overview

The backend for **AutoML Scientist** is built using Python 3.10+ and Flask. It serves as the orchestrator and execution environment for the autonomous ML research collaborator. Its primary responsibilities include:
1. Parsing and profiling uploaded tabular datasets.
2. Interacting with LLMs (Gemini for preprocessing code generation; Groq Llama-3 for orchestration).
3. Safely executing generated preprocessing and model training code in an isolated local sandbox.
4. Maintaining transient session state in-memory (no persistent database).
5. Streaming live research logs and experiment results to the React frontend via Server-Sent Events (SSE).

---

## Tech Stack

| Layer / Library | Choice | Purpose |
|---|---|---|
| **Framework** | Flask 3.0+ | Lightweight API server & routing |
| **CORS** | Flask-CORS 4.0+ | Cross-origin resource sharing for Vite React frontend |
| **Data Manipulation** | Pandas 2.0+, NumPy 1.24+ | Tabular profiling, cleaning, and preprocessing |
| **Machine Learning** | Scikit-learn 1.3+, Imbalanced-learn 0.12+ | Model training, CV evaluation, and metrics scoring |
| **LLM APIs** | Google GenAI SDK, Groq SDK | Prompting LLMs for code generation and decision making |
| **Serialization** | Joblib | Saving the best trained models |

---

## Repository Structure

```text
backend/
├── app.py                  # Main Flask Entrypoint (registers Blueprints, configures CORS)
├── config.py               # Configuration loading from environment/.env
├── memory_store.py         # Transient, in-memory session state management
├── requirements.txt        # Python dependency manifest
│
├── routers/                # Thin API routing layer
│   ├── __init__.py
│   ├── upload_router.py      # POST /api/upload
│   ├── codegen_router.py     # POST /api/codegen/<session_id>
│   └── orchestrate_router.py # GET /api/orchestrate/<session_id>
│
├── handlers/               # Route handling business logic controller
│   ├── __init__.py
│   ├── upload_handler.py     # Parses files, runs profiler, stores state
│   ├── codegen_handler.py    # Runs Gemini LLM client, executes clean code with sandbox
│   └── orchestrate_handler.py# Orchestrates research loop generator and executes model training
│
└── services/               # Core utility services
    ├── __init__.py
    ├── profiler.py           # Statistical profiling and problem type detection
    ├── gemini_api.py         # Low-level Gemini Client wrapper
    ├── llm_client.py         # High-level data cleaning prompt builder & corrector
    ├── groq_api.py           # Low-level Groq Client wrapper
    ├── code_generator.py     # Coordinates Gemini calls and retry loop for cleaning code
    ├── executor.py           # Sandboxed code execution (restricts built-ins, imports)
    ├── ml_engine.py          # Anchoring evaluations, CV scoring, and dataset intelligence compilation
    └── orchestrator.py       # Master prompt definition and autonomous research loop generator
```

---

## In-Memory Session State Schema

State is maintained in-memory in `memory_store.py` using standard dataclasses and a global `session_store` dictionary indexed by a unique `session_id` UUID:

```python
@dataclass
class DatasetState:
    session_id: str
    filename: str
    df_raw: pd.DataFrame       # Unmodified dataframe
    df_clean: pd.DataFrame     # Preprocessed dataframe
    problem_type: str          # "classification" | "regression"
    target_column: str
    summary: Dict[str, Any]    # Profile statistics & cardinality info

@dataclass
class ExperimentState:
    run_number: int
    model_name: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    code_snippet: str
    status: str                # "Success" | "Failed"
    reason: str                # LLM reasoning / diagnosis

@dataclass
class SessionState:
    dataset: DatasetState = None
    experiments: List[ExperimentState] = field(default_factory=list)
    narration_logs: List[str] = field(default_factory=list)
    best_score: float = -float("inf")
```

---

## Key Backend Workflows & Integration Points

### 1. Upload & Profile (`POST /api/upload`)
- Receives a tabular CSV dataset.
- Detects the problem type (classification vs. regression) based on target variable cardinality:
  - Numeric target with $> 10$ unique values $\rightarrow$ **Regression**
  - Text target or numeric target with $\le 10$ unique values $\rightarrow$ **Classification**
- Evaluates class distributions, missing values, correlations, and cardinalities.

### 2. Preprocessing Code Generation (`POST /api/codegen/<session_id>`)
- Compiles dataset metadata into a prompt for Gemini.
- Receives a Python function `def clean(df: pd.DataFrame) -> pd.DataFrame`.
- Executes this function in the sandboxed python namespace.
- If it fails, runs a self-correction retry loop (up to 2 times) with the error traceback, before falling back to a static scikit-learn pipeline.

### 3. Autonomous Research Loop (`GET /api/orchestrate/<session_id>`)
- Streams progress line-by-line using Server-Sent Events (SSE).
- Compiles the **Dataset Intelligence JSON** containing baseline scores (Logistic Regression / Decision Tree).
- Runs the orchestrator loop:
  1. Sends the Master Prompt (with previous history) to Groq (Llama-3.3-70b-versatile).
  2. Receives recommendations and training code string defining `def train(df, target_col)`.
  3. Executes the training code on the cleaned dataset.
  4. Returns the K-Fold CV score and trained model.
  5. Repeats (default: 2 iterations).
- Saves the best model locally in the `models/` directory using joblib.

### 4. Download Model (`GET /api/download/<session_id>`)
- Serves the best-performing model serialized `.pkl` file for local use.
