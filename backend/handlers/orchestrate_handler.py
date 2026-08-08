"""
Orchestrate Handler — Manages Journey F (Autonomous Research Loop).

Responsibilities:
    - Set up SSE generator for streaming logs
    - Pull DatasetState from memory
    - Compile Dataset Intelligence JSON
    - Connect to LLM Orchestrator
    - Execute training code and feed back to Orchestrator
"""
import json
import logging
from typing import Generator

# ── Safe Sklearn Monkey-Patching for Small Datasets ──────────────────────────
try:
    import numpy as np
    import sklearn.model_selection
    
    _orig_cross_val_score = sklearn.model_selection.cross_val_score
    _orig_cross_validate = sklearn.model_selection.cross_validate

    def safe_cross_val_score(*args, **kwargs):
        cv = kwargs.get("cv", 5)
        X = kwargs.get("X", None)
        if X is None and len(args) > 1:
            X = args[1]
        
        if X is not None:
            try:
                n_samples = X.shape[0]
                if isinstance(cv, int):
                    if n_samples < 2:
                        return np.array([1.0])
                    kwargs["cv"] = min(cv, n_samples)
            except Exception:
                pass
        try:
            return _orig_cross_val_score(*args, **kwargs)
        except ValueError as e:
            if "n_splits" in str(e) or "splits" in str(e):
                estimator = kwargs.get("estimator") or args[0]
                y = kwargs.get("y")
                if y is None and len(args) > 2:
                    y = args[2]
                estimator.fit(X, y)
                score = estimator.score(X, y)
                return np.array([score])
            raise e

    def safe_cross_validate(*args, **kwargs):
        cv = kwargs.get("cv", 5)
        X = kwargs.get("X", None)
        if X is None and len(args) > 1:
            X = args[1]
        
        if X is not None:
            try:
                n_samples = X.shape[0]
                if isinstance(cv, int):
                    if n_samples < 2:
                        kwargs["cv"] = 2
                    else:
                        kwargs["cv"] = min(cv, n_samples)
            except Exception:
                pass
        try:
            return _orig_cross_validate(*args, **kwargs)
        except ValueError as e:
            if "n_splits" in str(e) or "splits" in str(e):
                estimator = kwargs.get("estimator") or args[0]
                y = kwargs.get("y")
                if y is None and len(args) > 2:
                    y = args[2]
                estimator.fit(X, y)
                score = estimator.score(X, y)
                return {"test_score": np.array([score])}
            raise e

    sklearn.model_selection.cross_val_score = safe_cross_val_score
    sklearn.model_selection.cross_validate = safe_cross_validate
except Exception as ex:
    pass

from memory_store import get_session, ExperimentState
from services.ml_engine import compile_dataset_intelligence
from services.orchestrator import iterate_research_loop
from services.groq_api import check_groq_api_status

logger = logging.getLogger(__name__)


def format_sse(data: dict) -> str:
    """Format dictionary as Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"


def orchestrate_pipeline_stream(session_id: str) -> Generator[str, None, None]:
    """
    Generator that orchestrates the AI model research loop and yields SSE strings.
    """
    session = get_session(session_id)
    if not session:
        yield format_sse({
            "status": "error",
            "message": "Session not found."
        })
        return

    ds_state = session.dataset
    if not ds_state or ds_state.df_clean is None:
        yield format_sse({
            "status": "error",
            "message": "Dataset not found or not cleaned yet. Run codegen first."
        })
        return

    # Compile the intelligence JSON
    yield format_sse({
        "status": "in_progress",
        "phase": "Initialization",
        "message": "Evaluating local baselines and compiling Dataset Intelligence..."
    })

    problem_info = {
        "problem_type": ds_state.problem_type,
        "target_column": ds_state.target_column,
        "class_distribution": ds_state.summary.get("class_distribution", {}),
        "target_stats": ds_state.summary.get("target_stats", {})
    }

    # Convert session.experiments to serializable list for the LLM
    experiment_history = [
        {
            "iteration": exp.run_number,
            "model": exp.model_name,
            "score": exp.metrics.get("score", 0.0),
            "diagnosis": exp.reason,
            "result": exp.status
        }
        for exp in session.experiments
    ]

    try:
        dataset_intelligence = compile_dataset_intelligence(
            session_id=session_id,
            filename=ds_state.filename,
            df_raw=ds_state.df_raw,
            df_clean=ds_state.df_clean,
            profile=ds_state.summary,
            problem_info=problem_info,
            experiment_history=experiment_history
        )
    except Exception as e:
        logger.exception("Failed to compile dataset intelligence")
        yield format_sse({
            "status": "error",
            "message": f"Failed to compile dataset intelligence: {e}"
        })
        return

    # Confirm Groq API key status
    try:
        api_status = check_groq_api_status()
        if api_status["status"] == "working":
            logger.info(">>> [API STATUS] Groq API: WORKING - Successfully validated connection.")
            yield format_sse({
                "status": "in_progress",
                "phase": "Initialization",
                "message": "Groq API Status: WORKING - Successfully validated connection."
            })
        elif api_status["status"] == "inactive":
            logger.info(">>> [API STATUS] Groq API: INACTIVE (No API Key) - Falling back to local research engine.")
            yield format_sse({
                "status": "in_progress",
                "phase": "Initialization",
                "message": "Groq API Status: INACTIVE (No API Key) - Falling back to local research engine."
            })
        else:
            logger.info(f">>> [API STATUS] Groq API: ERROR ({api_status['message']}) - Falling back to local research engine.")
            yield format_sse({
                "status": "in_progress",
                "phase": "Initialization",
                "message": f"Groq API Status: ERROR ({api_status['message']}) - Falling back to local research engine."
            })
    except Exception as e:
        logger.warning(f"Failed to check Groq API status: {e}")

    # Start the LLM Orchestrator
    orchestrator_gen = iterate_research_loop(dataset_intelligence, max_iterations=2)
    
    try:
        # Start generator
        event = next(orchestrator_gen)
        
        while True:
            if event.get("status") == "experiment_proposed":
                # The orchestrator yielded a proposed experiment. We need to execute it.
                iteration = event["iteration"]
                llm_json = event["llm_json"]
                training_code = event["training_code"]
                
                yield format_sse({
                    "status": "in_progress",
                    "phase": f"Iteration {iteration}",
                    "message": f"Training code generated for {llm_json['recommended_pipeline']['recommended_model']}. Executing locally..."
                })
                
                # Execute training code
                local_ns = {
                    "df": ds_state.df_clean,
                    "target_col": ds_state.target_column
                }
                
                import sys
                import traceback
                import joblib
                import os
                
                execution_result = {}
                try:
                    import pandas as pd
                    orig_read_csv = pd.read_csv
                    try:
                        pd.read_csv = lambda *args, **kwargs: ds_state.df_clean
                        exec(training_code, local_ns)
                    finally:
                        pd.read_csv = orig_read_csv
                    train_func = local_ns.get("train")
                    if not train_func:
                        # Defensive extraction from local namespace if LLM forgot def train()
                        model = None
                        for k, v in local_ns.items():
                            if k in ["model", "clf", "estimator", "xgb", "rf", "gb", "lr", "gbm", "pipeline"]:
                                model = v
                                break
                        if not model:
                            for k, v in local_ns.items():
                                if hasattr(v, "fit") and hasattr(v, "predict"):
                                    model = v
                                    break
                                    
                        score = None
                        for k, v in local_ns.items():
                            if k in ["score", "cv_score", "accuracy", "acc", "val_score", "f1_score", "f1"]:
                                try:
                                    score = float(v)
                                    break
                                except (ValueError, TypeError):
                                    pass
                        if score is None:
                            if model is not None:
                                try:
                                    X = ds_state.df_clean.drop(columns=[ds_state.target_column])
                                    y = ds_state.df_clean[ds_state.target_column]
                                    score = float(model.score(X, y))
                                except Exception:
                                    score = 0.5
                            else:
                                score = 0.5
                                
                        execution_result = {"score": score}
                        metrics_payload = {"score": score}
                        gemini_prompt = None
                        if model is not None:
                            computed = compute_evaluation_metrics(session_id, model, ds_state.df_clean, ds_state.target_column, problem_info["problem_type"])
                            if "error" not in computed:
                                metrics_payload.update(computed)
                    else:
                        result_tuple = train_func(ds_state.df_clean, ds_state.target_column)
                        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                            score, model = result_tuple
                        elif isinstance(result_tuple, dict):
                            score = float(result_tuple.get("score", result_tuple.get("cv_score", 0.0)))
                            model = result_tuple.get("model", result_tuple.get("estimator", None))
                        else:
                            try:
                                score = float(result_tuple)
                                model = None
                            except (ValueError, TypeError) as ex:
                                score = 0.0
                                model = None
                            
                        execution_result = {"score": score}
                        metrics_payload = {"score": score}
                        gemini_prompt = None
                        if model is not None:
                            computed = compute_evaluation_metrics(session_id, model, ds_state.df_clean, ds_state.target_column, problem_info["problem_type"])
                            if "error" not in computed:
                                metrics_payload.update(computed)
                                
                            from services.gemini_api import call_gemini, is_api_available
                            if is_api_available():
                                logger.info(">>> [COLLABORATION] Invoking collaborative Gemini layer to review model performance...")
                                yield format_sse({
                                    "status": "in_progress",
                                    "phase": f"Iteration {iteration}",
                                    "message": f"Invoking collaborative Gemini layer to review model performance and prepare the next prompt..."
                                })
                                metrics = computed
                                
                                prompt_to_gemini = f"""
You are an expert ML Scientific Reviewer and Consultant.
Analyze the following empirical model performance results from our local training sandbox.

## Run Information
- Iteration: {iteration}
- Model trained: {llm_json['recommended_pipeline']['recommended_model']}
- Hyperparameters used: {json.dumps(llm_json['recommended_pipeline'].get('hyperparameters', {}))}

## Dataset Details
- Problem Type: {problem_info['problem_type']}
- Target Column: {ds_state.target_column}

## Local Evaluation Metrics (Full Dataset):
{json.dumps(metrics, indent=2)}

## Preprocessing & Model Code Executed:
```python
{training_code}
```

## TASK:
1. Formulate a comprehensive scientific diagnosis of why the model performed this way (e.g. check for overfitting/underfitting, check classification report metrics per class, check correlation or outliers).
2. Propose concrete hyperparameter changes, algorithm switches, or preprocessing fixes to optimize results.
3. Construct the EXACT MASTER PROMPT text that should be sent to the Groq code generator (Llama-3) in the next iteration. 
   - This prompt MUST instruct Groq on what model family, hyperparameter ranges, and fixes to generate in the next iteration.
   - It MUST contain the updated Dataset Intelligence JSON with the new experiment history.
   - The prompt MUST tell Groq to output ONLY the required JSON containing the training code.
   - Keep the prompt professional, dense, and structured.

Return a valid JSON object matching this schema:
{{
  "diagnosis": "...",
  "suggestions": ["...", "..."],
  "next_prompt_for_groq": "..."
}}
"""
                                gemini_response = call_gemini(prompt_to_gemini, temperature=0.2)
                                if not gemini_response["error"]:
                                    try:
                                        raw_text = gemini_response["text"].strip()
                                        if raw_text.startswith("```json"):
                                            raw_text = raw_text[7:]
                                        if raw_text.endswith("```"):
                                            raw_text = raw_text[:-3]
                                        parsed = json.loads(raw_text.strip())
                                        gemini_prompt = parsed.get("next_prompt_for_groq")
                                        
                                        logger.info(f">>> [COLLABORATION] Gemini review complete. Suggestions generated: {', '.join(parsed.get('suggestions', []))[:120]}...")
                                        yield format_sse({
                                            "status": "in_progress",
                                            "phase": f"Iteration {iteration}",
                                            "message": f"Gemini review complete. Suggestions generated: {', '.join(parsed.get('suggestions', []))[:120]}..."
                                        })
                                    except Exception as ex:
                                        logger.warning(f"Failed to parse Gemini response: {ex}")
                                else:
                                    logger.warning(f">>> [COLLABORATION] Gemini API returned error: {gemini_response['error']}")
                                    yield format_sse({
                                        "status": "in_progress",
                                        "phase": f"Iteration {iteration}",
                                        "message": f"Gemini layer limits exceeded ({gemini_response['error'][:60]}...). Continuing research flow..."
                                    })
                            else:
                                logger.info(">>> [COLLABORATION] Gemini API Key not configured. Skipping Gemini reviewer layer.")
                                    
                        execution_result["gemini_prompt"] = gemini_prompt

                        
                        # Save to session memory
                        exp = ExperimentState(
                            run_number=iteration,
                            model_name=llm_json['recommended_pipeline']['recommended_model'],
                            hyperparameters=llm_json['recommended_pipeline'].get('hyperparameters', {}),
                            metrics=metrics_payload,
                            code_snippet=training_code,
                            status="Success",
                            reason=llm_json['diagnosis']['reasoning']
                        )
                        session.experiments.append(exp)
                        
                        # Track the best model and save it
                        best_score = getattr(session, "best_score", -float("inf"))
                        if score > best_score and model is not None:
                            session.best_score = score
                            os.makedirs("models", exist_ok=True)
                            model_path = os.path.join("models", f"{session_id}_best_model.pkl")
                            joblib.dump(model, model_path)
                            execution_result["best_model_path"] = model_path
                        
                except Exception as e:
                    execution_result = {"error": str(e) + "\\n" + traceback.format_exc()}
                
                # Send the execution result back into the generator
                event = orchestrator_gen.send(execution_result)
            else:
                # Forward standard status/progress events to the client
                session.narration_logs.append(event.get("message", ""))
                yield format_sse(event)
                # Get next event
                event = next(orchestrator_gen)

    except StopIteration:
        # Generator finished
        pass
    except Exception as e:
        logger.exception("Error in research loop")
        yield format_sse({
            "status": "error",
            "message": f"Orchestration failed: {e}"
        })


def compute_evaluation_metrics(session_id, model, df, target_col, problem_type):
    from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    try:
        X = df.drop(columns=[target_col])
        y_true = df[target_col]
        y_pred = model.predict(X)
        
        # Extract feature importances if available
        importances = {}
        features = X.columns.tolist()
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(features, [float(v) for v in model.feature_importances_]))
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                coef_vals = np.abs(coefs).mean(axis=0).tolist()
            else:
                coef_vals = np.abs(coefs).tolist()
            importances = dict(zip(features, [float(v) for v in coef_vals]))
        
        # ── Generate and save Feature Importance plot ──
        if importances:
            try:
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
                if sorted_imp:
                    imp_features = [x[0] for x in sorted_imp]
                    imp_values = [x[1] for x in sorted_imp]
                    plt.figure(figsize=(6, 4))
                    colors = sns.color_palette("viridis", len(imp_features))
                    plt.barh(imp_features[::-1], imp_values[::-1], color=colors)
                    plt.xlabel("Relative Importance Score", fontsize=9)
                    plt.title("Feature Importance Analysis (Best Model)", fontsize=10, fontweight="bold", pad=15)
                    plt.tight_layout()
                    
                    os.makedirs(os.path.join("static", "plots"), exist_ok=True)
                    plt.savefig(os.path.join("static", "plots", f"{session_id}_importance.png"), dpi=150)
                    plt.close()
            except Exception as ex:
                print(f"Failed to generate importance plot: {ex}")

        if problem_type == "classification":
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            acc = report.get("accuracy", 0.0)
            
            # ── Generate and save Confusion Matrix plot ──
            try:
                plt.figure(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Greens",
                    cbar=False,
                    square=True,
                    annot_kws={"size": 14, "weight": "bold"}
                )
                plt.xlabel("Predicted Class", fontsize=9, labelpad=8)
                plt.ylabel("True Class", fontsize=9, labelpad=8)
                plt.title("Confusion Matrix (Best Model)", fontsize=10, fontweight="bold", pad=15)
                plt.tight_layout()
                
                os.makedirs(os.path.join("static", "plots"), exist_ok=True)
                plt.savefig(os.path.join("static", "plots", f"{session_id}_confusion.png"), dpi=150)
                plt.close()
            except Exception as ex:
                print(f"Failed to generate confusion plot: {ex}")

            return {
                "accuracy": acc,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "feature_importances": importances,
                "summary": f"Accuracy: {acc:.4f}, Macro F1: {report.get('macro avg', {}).get('f1-score', 0.0):.4f}"
            }
        else:
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            return {
                "r2_score": r2,
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
                "feature_importances": importances,
                "summary": f"R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
            }
    except Exception as e:
        return {"error": f"Failed to compute validation metrics: {str(e)}"}

