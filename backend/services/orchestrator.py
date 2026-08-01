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
    "training_code": "import pandas as pd\\nimport numpy as np\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.model_selection import cross_val_score\\n\\ndef train(df, target_col):\\n    X = df.drop(columns=[target_col])\\n    y = df[target_col]\\n    model = RandomForestClassifier(n_estimators=100)\\n    scores = cross_val_score(model, X, y, cv=5)\\n    return float(scores.mean())"
  }},
  "explanation": {{
    "technical": "..."
  }}
}}

NOTE ON training_code: 
Your `training_code` string MUST contain a Python function `def train(df, target_col):` that returns a single float (the validation score).
It must import all necessary sklearn libraries.
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

    history = dataset_intelligence.get("experiment_history", [])

    for iteration in range(1, max_iterations + 1):
        yield {
            "status": "in_progress",
            "phase": f"Iteration {iteration}",
            "message": f"Analyzing dataset intelligence (Iteration {iteration}/{max_iterations})..."
        }

        # Update iteration count in intelligence
        dataset_intelligence["session"]["experiment_iteration"] = iteration
        dataset_intelligence["experiment_history"] = history

        prompt = MASTER_PROMPT.format(
            dataset_intelligence=json.dumps(dataset_intelligence, indent=2)
        )

        response = call_groq_json(prompt)
        
        if response["error"]:
            yield {
                "status": "error",
                "phase": f"Iteration {iteration}",
                "message": f"LLM Error: {response['error']}"
            }
            break
            
        llm_json = response["json"]
        
        yield {
            "status": "in_progress",
            "phase": f"Iteration {iteration}",
            "message": f"LLM recommends: {llm_json['recommended_pipeline']['recommended_model']}. Reasoning: {llm_json['diagnosis']['primary_issue']}"
        }

        training_code = llm_json["code_changes"].get("training_code", "")
        
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

        # We will use the executor but look for 'train' instead of 'clean'
        # The executor needs to be slightly modified or we write a local runner for 'train'
        # To avoid modifying executor too much, we'll execute it locally here for Journey E
        import sys
        import io
        
        local_ns = {}
        global_ns = {
            "__builtins__": __builtins__
        }
        
        try:
            exec(training_code, global_ns)
            train_func = global_ns.get("train")
            if not train_func:
                raise ValueError("Generated code does not define a 'train(df, target_col)' function.")
            
            # Note: We need df_clean and target_column here. 
            # We'll expect the handler to have put them in the generator context.
            # However, since this is a generator, we actually need to pass df_clean here.
            # For simplicity, we'll just yield the code back to the handler and let the handler execute it.
            
        except Exception as e:
            # Yield error back
            pass

        # Since this generator doesn't have df_clean directly, I will refactor to have the handler call
        # the ML engine. Actually, let's just make the orchestrator return the LLM response, and the handler does the execution.
        
        # We will yield the LLM proposed experiment and wait for the execution result
        execution_result = yield {
            "status": "experiment_proposed",
            "llm_json": llm_json,
            "training_code": training_code,
            "iteration": iteration
        }
        
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
                "status": "in_progress",
                "phase": f"Iteration {iteration}",
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
