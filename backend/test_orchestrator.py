import os
import sys
import pandas as pd
import numpy as np

# Ensure backend is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_store import create_session, get_session, DatasetState
from handlers.orchestrate_handler import orchestrate_pipeline_stream

def create_synthetic_clean_dataset():
    """Create a clean classification dataset for testing."""
    np.random.seed(42)
    
    # 200 rows, 3 features
    age = np.random.normal(30, 10, 200)
    salary = np.random.normal(50000, 15000, 200)
    score = np.random.uniform(0, 100, 200)
    
    # Target: 1 if salary > 50k and score > 50, else 0
    target = ((salary > 50000) & (score > 50)).astype(int)
    
    df = pd.DataFrame({
        "Age": age,
        "Salary": salary,
        "Score": score,
        "Target": target
    })
    
    return df

def run_test():
    print("==================================================")
    print("  TESTING AUTONOMOUS RESEARCH LOOP (JOURNEY E/F)  ")
    print("==================================================")
    
    session_id = create_session()
    session = get_session(session_id)
    
    df_clean = create_synthetic_clean_dataset()
    
    # Mock the DatasetState
    session.dataset = DatasetState(
        session_id=session_id,
        filename="test_clean.csv",
        df_raw=df_clean.copy(),
        df_clean=df_clean,
        problem_type="classification",
        target_column="Target",
        summary={
            "missing_values": {},
            "basic_stats": {},
            "class_distribution": {"0": 150, "1": 50},
            "target_stats": {"entropy": 0.5}
        }
    )
    
    print(f"Created mocked session {session_id} with cleaned dataset.")
    print("Starting SSE Stream...\n")
    
    generator = orchestrate_pipeline_stream(session_id)
    
    for sse_event in generator:
        # The generator yields strings like "data: {...}\n\n"
        if sse_event.startswith("data: "):
            try:
                import json
                data_str = sse_event[6:].strip()
                data = json.loads(data_str)
                
                status = data.get("status")
                phase = data.get("phase")
                msg = data.get("message")
                
                if status == "error":
                    safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
                    print(f"[ERROR] {phase}: {safe_msg}")
                elif status == "done":
                    safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
                    print(f"[DONE] {safe_msg}")
                else:
                    safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
                    print(f"[{phase}] {safe_msg}")
                    
            except Exception as e:
                safe_event = str(sse_event).encode('ascii', errors='replace').decode('ascii')
                print(f"Failed to parse SSE event: {safe_event}")
                print(e)
                
    print("\n==================================================")
    print("  TEST COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_test()
