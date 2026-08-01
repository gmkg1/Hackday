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

from memory_store import get_session, ExperimentState
from services.ml_engine import compile_dataset_intelligence
from services.orchestrator import iterate_research_loop


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
                
                execution_result = {}
                try:
                    exec(training_code, local_ns)
                    train_func = local_ns.get("train")
                    if not train_func:
                        execution_result = {"error": "No 'train' function defined."}
                    else:
                        score = train_func(ds_state.df_clean, ds_state.target_column)
                        execution_result = {"score": score}
                        
                        # Save to session memory
                        exp = ExperimentState(
                            run_number=iteration,
                            model_name=llm_json['recommended_pipeline']['recommended_model'],
                            hyperparameters=llm_json['recommended_pipeline'].get('hyperparameters', {}),
                            metrics={"score": score},
                            code_snippet=training_code,
                            status="Success",
                            reason=llm_json['diagnosis']['reasoning']
                        )
                        session.experiments.append(exp)
                        
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
