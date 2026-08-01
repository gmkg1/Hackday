"""
LLM Orchestrator Service — Journey F (Autonomous Research Loop).

Responsibilities:
    - Manage the Master LLM Prompt
    - Interact with Groq API (JSON mode)
    - Run the autonomous ML research loop
"""
import json
import logging
from typing import Dict, Any, Generator

from services.groq_api import call_groq_json
from services.executor import run_generated_code


# ==============================================================================
#                      MASTER LLM ORCHESTRATOR PROMPT
# ==============================================================================
# This prompt guides the autonomous ML research loop (Journey F).
# It instructs the LLM (typically Llama-3-70b via Groq) to act as a seasoned
# data scientist.
#
# Key features of this prompt:
#   1. Input: Dataset Intelligence JSON (schema, metrics, history, baselines).
#   2. Output: Must return a STRICT JSON matching the schema containing:
#      - "diagnosis": explanation of dataset weaknesses.
#      - "recommended_pipeline": recommended model family and hyperparameters.
#      - "code_changes": Python code containing a `def train(df, target_col):` function.
#      - "explanation": technical rationale.
#   3. Model Selection rules: Instructs the LLM to vary its selections over 
#      iterations, avoid code repetition (especially RandomForest), and match 
#      the algorithm to dataset complexity.
#   4. Validation instructions: Reminds the LLM of libraries/parameters to import
#      and not to use class_weight for GradientBoosting estimators.
# ==============================================================================
MASTER_PROMPT = """\
You are AutoML Scientist.
You are an autonomous Machine Learning Research Scientist.
Your purpose is NOT simply to maximize accuracy.
Your purpose is to think like an experienced ML researcher.

You are given:
• Complete dataset metadata
• Statistical analysis
• Dataset complexity metrics
• Baseline model performances
• Previous experiment history

You must carefully analyze everything before making any recommendation.
Never assume.
Base every decision on evidence.

--------------------------------------------------
OBJECTIVES
1. Understand the dataset.
2. Identify strengths and weaknesses of the current preprocessing.
3. Determine whether the chosen model family is appropriate.
4. Diagnose why the current pipeline produced its results.
5. Recommend concrete improvements.
6. Rewrite training code to evaluate your recommendation.
7. Explain every recommendation.

--------------------------------------------------
DATASET INTELLIGENCE JSON:
{dataset_intelligence}

--------------------------------------------------
Your diagnosis should consider:
• Class imbalance
• Multicollinearity
• Outliers
• Non-linearity
• Overfitting vs Underfitting
• Previous experiment history

Rules of Engagement:
• Never recommend a model without explaining WHY.
• Never repeat an experiment that already failed unless there is a significant modification.
• Do NOT simply output the same model family (e.g., RandomForest) repeatedly. You must select a model family appropriate to the dataset characteristics (e.g., regularized linear models for simple/linear datasets, tree-based ensembles like GradientBoostingClassifier, XGBoostClassifier, RandomForestClassifier, DecisionTreeClassifier for complex/non-linear datasets).
• Return ONLY valid JSON matching this schema exactly:

{{
  "summary": {{
    "dataset_type": "binary_classification",
    "difficulty": "Medium",
    "confidence": 0.88
  }},
  "diagnosis": {{
    "primary_issue": "...",
    "secondary_issues": ["..."],
    "reasoning": "..."
  }},
  "recommended_pipeline": {{
    "recommended_model": "...",
    "hyperparameters": {{}}
  }},
  "code_changes": {{
    "training_code": "import pandas as pd\\nimport numpy as np\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.model_selection import cross_val_score\\n\\ndef train(df, target_col):\\n    X = df.drop(columns=[target_col])\\n    y = df[target_col]\\n    model = RandomForestClassifier(n_estimators=100)\\n    scores = cross_val_score(model, X, y, cv=5)\\n    # Retrain on full data to return the final model\\n    model.fit(X, y)\\n    return float(scores.mean()), model"
  }},
  "explanation": {{
    "technical": "..."
  }}
}}

NOTE ON training_code: 
Your `training_code` string MUST contain a Python function `def train(df, target_col):` that returns a tuple `(score, model)`.
The `score` MUST be a single float (the validation score).
The `model` MUST be the trained scikit-learn model object (ready for predictions).
It must import all necessary sklearn libraries.
CRITICAL: Note that GradientBoostingClassifier, GradientBoostingRegressor, XGBClassifier, and XGBRegressor do NOT accept a 'class_weight' parameter in scikit-learn or xgboost. Do not pass 'class_weight' to them!
"""


SCHEMA_INSTRUCTIONS = """
Rules of Engagement:
• Never recommend a model without explaining WHY.
• Never repeat an experiment that already failed unless there is a significant modification.
• Do NOT simply output the same model family (e.g., RandomForest) repeatedly. You must select a model family appropriate to the dataset characteristics (e.g., regularized linear models for simple/linear datasets, tree-based ensembles like GradientBoostingClassifier, XGBoostClassifier, RandomForestClassifier, DecisionTreeClassifier for complex/non-linear datasets).
• Return ONLY valid JSON matching this schema exactly:

{
  "summary": {
    "dataset_type": "binary_classification",
    "difficulty": "Medium",
    "confidence": 0.88
  },
  "diagnosis": {
    "primary_issue": "...",
    "secondary_issues": ["..."],
    "reasoning": "..."
  },
  "recommended_pipeline": {
    "recommended_model": "...",
    "hyperparameters": {}
  },
  "code_changes": {
    "training_code": "import pandas as pd\\nimport numpy as np\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.model_selection import cross_val_score\\n\\ndef train(df, target_col):\\n    X = df.drop(columns=[target_col])\\n    y = df[target_col]\\n    model = RandomForestClassifier(n_estimators=100)\\n    scores = cross_val_score(model, X, y, cv=5)\\n    # Retrain on full data to return the final model\\n    model.fit(X, y)\\n    return float(scores.mean()), model"
  },
  "explanation": {
    "technical": "..."
  }
}

NOTE ON training_code: 
Your `training_code` string MUST contain a Python function `def train(df, target_col):` that returns a tuple `(score, model)`.
The `score` MUST be a single float (the validation score).
The `model` MUST be the trained scikit-learn model object (ready for predictions).
It must import all necessary sklearn libraries.
CRITICAL: Note that GradientBoostingClassifier, GradientBoostingRegressor, XGBClassifier, and XGBRegressor do NOT accept a 'class_weight' parameter in scikit-learn or xgboost. Do not pass 'class_weight' to them!
"""


def iterate_research_loop(
    dataset_intelligence: Dict[str, Any],
    max_iterations: int = 2
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that orchestrates the autonomous research loop.
    Yields event dictionaries that the handler will stream to the client via SSE.
    """
    
    yield {
        "status": "in_progress",
        "phase": "Orchestrator",
        "message": "Starting Autonomous Research Loop..."
    }

    # Determine if we should use fallback mock logic
    from services.groq_api import is_groq_available, check_groq_api_status
    use_fallback = not is_groq_available()
    if not use_fallback:
        try:
            api_status = check_groq_api_status()
            if api_status["status"] != "working":
                use_fallback = True
        except Exception:
            use_fallback = True

    if use_fallback:
        yield {
            "status": "in_progress",
            "phase": "Orchestrator",
            "message": "Groq/OpenRouter API connection failed or key not configured. Initiating deterministic local research loop..."
        }

    history = dataset_intelligence.get("experiment_history", [])
    problem_type = dataset_intelligence.get("dataset", {}).get("problem_type", "classification")
    last_gemini_prompt = None

    for iteration in range(1, max_iterations + 1):
        yield {
            "status": "in_progress",
            "phase": f"Iteration {iteration}",
            "message": f"Analyzing dataset intelligence (Iteration {iteration}/{max_iterations})..."
        }

        # Update iteration count in intelligence
        dataset_intelligence["session"]["experiment_iteration"] = iteration
        dataset_intelligence["experiment_history"] = history

        if use_fallback:
            logger = logging.getLogger("app")
            logger.info(f">>>> [ORCHESTRATOR] Iteration {iteration}: Running in LOCAL BACKUP MODE (fallback engine).")
            # Deterministic mock pipeline training real models locally
            if iteration == 1:
                rec_model = "Random Forest Classifier" if problem_type == "classification" else "Random Forest Regressor"
                primary_issue = "Capturing non-linear interactions in cleaned dataset."
                reasoning = "Logistic / linear baselines might underfit. Upgrading to a Random Forest ensemble to establish a non-linear benchmark."
                technical_exp = f"Upgraded model to {'RandomForestClassifier' if problem_type == 'classification' else 'RandomForestRegressor'} to model non-linear boundaries. Configured 100 estimators with default hyperparameters to establish an ensemble baseline."
                
                if problem_type == "classification":
                    training_code = (
                        "import pandas as pd\n"
                        "import numpy as np\n"
                        "from sklearn.ensemble import RandomForestClassifier\n"
                        "from sklearn.model_selection import cross_val_score\n\n"
                        "def train(df, target_col):\n"
                        "    X = df.drop(columns=[target_col])\n"
                        "    y = df[target_col]\n"
                        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                        "    scores = cross_val_score(model, X, y, cv=5)\n"
                        "    model.fit(X, y)\n"
                        "    return float(scores.mean()), model"
                    )
                else:
                    training_code = (
                        "import pandas as pd\n"
                        "import numpy as np\n"
                        "from sklearn.ensemble import RandomForestRegressor\n"
                        "from sklearn.model_selection import cross_val_score\n\n"
                        "def train(df, target_col):\n"
                        "    X = df.drop(columns=[target_col])\n"
                        "    y = df[target_col]\n"
                        "    model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
                        "    scores = cross_val_score(model, X, y, cv=5)\n"
                        "    model.fit(X, y)\n"
                        "    return float(scores.mean()), model"
                    )
            else:
                rec_model = "Gradient Boosting Classifier" if problem_type == "classification" else "Gradient Boosting Regressor"
                primary_issue = "Further minimizing residuals via boosted trees."
                reasoning = "Ensemble performance from Random Forest can be improved by training weak learners sequentially using gradient descent."
                technical_exp = f"Transitioned to {'GradientBoostingClassifier' if problem_type == 'classification' else 'GradientBoostingRegressor'} to sequentially optimize loss. Applied standard depth regularization to prevent overfitting on the ensemble iterations."
                
                if problem_type == "classification":
                    training_code = (
                        "import pandas as pd\n"
                        "import numpy as np\n"
                        "from sklearn.ensemble import GradientBoostingClassifier\n"
                        "from sklearn.model_selection import cross_val_score\n\n"
                        "def train(df, target_col):\n"
                        "    X = df.drop(columns=[target_col])\n"
                        "    y = df[target_col]\n"
                        "    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n"
                        "    scores = cross_val_score(model, X, y, cv=5)\n"
                        "    model.fit(X, y)\n"
                        "    return float(scores.mean()), model"
                    )
                else:
                    training_code = (
                        "import pandas as pd\n"
                        "import numpy as np\n"
                        "from sklearn.ensemble import GradientBoostingRegressor\n"
                        "from sklearn.model_selection import cross_val_score\n\n"
                        "def train(df, target_col):\n"
                        "    X = df.drop(columns=[target_col])\n"
                        "    y = df[target_col]\n"
                        "    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n"
                        "    scores = cross_val_score(model, X, y, cv=5)\n"
                        "    model.fit(X, y)\n"
                        "    return float(scores.mean()), model"
                    )

            llm_json = {
                "summary": {
                    "dataset_type": "binary_classification" if problem_type == "classification" else "regression",
                    "difficulty": "Medium",
                    "confidence": 0.90
                },
                "diagnosis": {
                    "primary_issue": primary_issue,
                    "secondary_issues": [],
                    "reasoning": reasoning
                },
                "recommended_pipeline": {
                    "recommended_model": rec_model,
                    "hyperparameters": {"n_estimators": 100, "random_state": 42}
                },
                "code_changes": {
                    "training_code": training_code
                },
                "explanation": {
                    "technical": technical_exp
                }
            }
        else:
            # Live LLM loop using Groq
            logger = logging.getLogger("app")
            if last_gemini_prompt:
                logger.info(f">>>> [ORCHESTRATOR] Iteration {iteration}: Sending Gemini-engineered prompt to Groq API...")
                prompt = last_gemini_prompt
            else:
                logger.info(f">>>> [ORCHESTRATOR] Iteration {iteration}: Sending default Master Prompt to Groq API...")
                prompt = MASTER_PROMPT.format(
                    dataset_intelligence=json.dumps(dataset_intelligence, indent=2)
                )

            response = call_groq_json(prompt)
            
            if response["error"]:
                logger.error(f">>>> [ORCHESTRATOR] Iteration {iteration}: Groq API returned error: {response['error']}. Switching to fallback engine.")
                yield {
                    "status": "in_progress",
                    "phase": f"Iteration {iteration}",
                    "message": f"LLM Error ({response['error']}). Falling back to local research engine..."
                }
                use_fallback = True
                if iteration == 1:
                    rec_model = "Random Forest Classifier" if problem_type == "classification" else "Random Forest Regressor"
                    primary_issue = "Capturing non-linear interactions in cleaned dataset."
                    reasoning = "Logistic / linear baselines might underfit. Upgrading to a Random Forest ensemble to establish a non-linear benchmark."
                    technical_exp = f"Upgraded model to {'RandomForestClassifier' if problem_type == 'classification' else 'RandomForestRegressor'} to model non-linear boundaries. Configured 100 estimators with default hyperparameters to establish an ensemble baseline."
                    
                    if problem_type == "classification":
                        training_code = (
                            "import pandas as pd\n"
                            "import numpy as np\n"
                            "from sklearn.ensemble import RandomForestClassifier\n"
                            "from sklearn.model_selection import cross_val_score\n\n"
                            "def train(df, target_col):\n"
                            "    X = df.drop(columns=[target_col])\n"
                            "    y = df[target_col]\n"
                            "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                            "    scores = cross_val_score(model, X, y, cv=5)\n"
                            "    model.fit(X, y)\n"
                            "    return float(scores.mean()), model"
                        )
                    else:
                        training_code = (
                            "import pandas as pd\n"
                            "import numpy as np\n"
                            "from sklearn.ensemble import RandomForestRegressor\n"
                            "from sklearn.model_selection import cross_val_score\n\n"
                            "def train(df, target_col):\n"
                            "    X = df.drop(columns=[target_col])\n"
                            "    y = df[target_col]\n"
                            "    model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
                            "    scores = cross_val_score(model, X, y, cv=5)\n"
                            "    model.fit(X, y)\n"
                            "    return float(scores.mean()), model"
                        )
                else:
                    rec_model = "Gradient Boosting Classifier" if problem_type == "classification" else "Gradient Boosting Regressor"
                    primary_issue = "Further minimizing residuals via boosted trees."
                    reasoning = "Ensemble performance from Random Forest can be improved by training weak learners sequentially using gradient descent."
                    technical_exp = f"Transitioned to {'GradientBoostingClassifier' if problem_type == 'classification' else 'GradientBoostingRegressor'} to sequentially optimize loss. Applied standard depth regularization to prevent overfitting on the ensemble iterations."
                    
                    if problem_type == "classification":
                        training_code = (
                            "import pandas as pd\n"
                            "import numpy as np\n"
                            "from sklearn.ensemble import GradientBoostingClassifier\n"
                            "from sklearn.model_selection import cross_val_score\n\n"
                            "def train(df, target_col):\n"
                            "    X = df.drop(columns=[target_col])\n"
                            "    y = df[target_col]\n"
                            "    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n"
                            "    scores = cross_val_score(model, X, y, cv=5)\n"
                            "    model.fit(X, y)\n"
                            "    return float(scores.mean()), model"
                        )
                    else:
                        training_code = (
                            "import pandas as pd\n"
                            "import numpy as np\n"
                            "from sklearn.ensemble import GradientBoostingRegressor\n"
                            "from sklearn.model_selection import cross_val_score\n\n"
                            "def train(df, target_col):\n"
                            "    X = df.drop(columns=[target_col])\n"
                            "    y = df[target_col]\n"
                            "    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n"
                            "    scores = cross_val_score(model, X, y, cv=5)\n"
                            "    model.fit(X, y)\n"
                            "    return float(scores.mean()), model"
                        )

                llm_json = {
                    "summary": {
                        "dataset_type": "binary_classification" if problem_type == "classification" else "regression",
                        "difficulty": "Medium",
                        "confidence": 0.90
                    },
                    "diagnosis": {
                        "primary_issue": primary_issue,
                        "secondary_issues": [],
                        "reasoning": reasoning
                    },
                    "recommended_pipeline": {
                        "recommended_model": rec_model,
                        "hyperparameters": {"n_estimators": 100, "random_state": 42}
                    },
                    "code_changes": {
                        "training_code": training_code
                    },
                    "explanation": {
                        "technical": technical_exp
                    }
                }
            else:
                llm_json = response["json"]
                
                # Defensive parsing of Llama-3's JSON response
                if "code_changes" in llm_json and isinstance(llm_json["code_changes"], dict):
                    training_code = llm_json["code_changes"].get("training_code", "")
                elif "code" in llm_json:
                    training_code = llm_json["code"]
                elif "training_code" in llm_json:
                    training_code = llm_json["training_code"]
                else:
                    # Look for any key containing "code" or "train"
                    training_code = ""
                    for k, v in llm_json.items():
                        if "code" in k.lower() or "train" in k.lower():
                            if isinstance(v, str):
                                training_code = v
                                break
                            elif isinstance(v, dict) and "training_code" in v:
                                training_code = v["training_code"]
                                break

                # Defensively ensure other expected keys exist
                if "recommended_pipeline" not in llm_json:
                    llm_json["recommended_pipeline"] = {
                        "recommended_model": llm_json.get("recommended_model", "XGBoost Classifier"),
                        "hyperparameters": llm_json.get("hyperparameters", {})
                    }
                elif isinstance(llm_json["recommended_pipeline"], dict):
                    if "recommended_model" not in llm_json["recommended_pipeline"]:
                        llm_json["recommended_pipeline"]["recommended_model"] = llm_json.get("recommended_model", "XGBoost Classifier")
                
                if "diagnosis" not in llm_json:
                    llm_json["diagnosis"] = {
                        "primary_issue": llm_json.get("primary_issue", "Improving model configuration based on Gemini review."),
                        "reasoning": llm_json.get("reasoning", "Tuned hyperparameters sequentially.")
                    }
                elif isinstance(llm_json["diagnosis"], dict):
                    if "primary_issue" not in llm_json["diagnosis"]:
                        llm_json["diagnosis"]["primary_issue"] = llm_json.get("primary_issue", "Improving model configuration.")
                    if "reasoning" not in llm_json["diagnosis"]:
                        llm_json["diagnosis"]["reasoning"] = llm_json.get("reasoning", "Tuned hyperparameters sequentially.")

                if "explanation" not in llm_json:
                    llm_json["explanation"] = {
                        "technical": llm_json.get("explanation_text", llm_json.get("reasoning", "Model evaluated successfully."))
                    }
                elif isinstance(llm_json["explanation"], dict):
                    if "technical" not in llm_json["explanation"]:
                        llm_json["explanation"]["technical"] = llm_json.get("explanation_text", llm_json["explanation"].get("beginner", "Model evaluated successfully."))
                elif isinstance(llm_json["explanation"], str):
                    llm_json["explanation"] = {
                        "technical": llm_json["explanation"]
                    }



        yield {
            "status": "in_progress",
            "phase": f"Iteration {iteration}",
            "message": f"LLM recommends: {llm_json['recommended_pipeline']['recommended_model']}. Reasoning: {llm_json['diagnosis']['primary_issue']}"
        }
        
        if not training_code:
            yield {
                "status": "error",
                "phase": f"Iteration {iteration}",
                "message": "LLM failed to provide training code."
            }
            break

        yield {
            "status": "in_progress",
            "phase": f"Iteration {iteration}",
            "message": "Executing recommended model in sandbox..."
        }

        # Yield the proposed experiment to let the handler execute it
        execution_result = yield {
            "status": "experiment_proposed",
            "llm_json": llm_json,
            "training_code": training_code,
            "iteration": iteration
        }
        
        last_gemini_prompt = execution_result.get("gemini_prompt")
        
        if execution_result.get("error"):
            yield {
                "status": "in_progress",
                "phase": f"Iteration {iteration}",
                "message": f"Execution failed: {execution_result['error']}. Recording failure."
            }
            history.append({
                "iteration": iteration,
                "model": llm_json['recommended_pipeline']['recommended_model'],
                "score": 0.0,
                "diagnosis": llm_json['diagnosis']['reasoning'],
                "result": f"Failed: {execution_result['error']}"
            })
        else:
            score = execution_result.get("score", 0.0)
            yield {
                "status": "experiment_complete",
                "iteration": iteration,
                "model_name": llm_json['recommended_pipeline']['recommended_model'],
                "metrics": {"score": score},
                "explanation": llm_json['explanation'].get('technical', 'Model evaluated successfully.'),
                "message": f"Execution succeeded! Score: {score:.4f}"
            }
            history.append({
                "iteration": iteration,
                "model": llm_json['recommended_pipeline']['recommended_model'],
                "score": score,
                "diagnosis": llm_json['diagnosis']['reasoning'],
                "result": "Success"
            })
            
    yield {
        "status": "done",
        "phase": "Completed",
        "message": "Autonomous Research Loop has completed."
    }

