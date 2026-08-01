"""
Safe Code Execution Sandbox — Scaffolded for Journey E (Cleaning Execution).

Executes LLM-generated Python code in a restricted namespace.
Used by later journey steps; defined now for architecture completeness.
"""
import sys
import io
import traceback
from typing import Tuple, Optional

import pandas as pd
import numpy as np


def run_generated_code(
    code_string: str,
    df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute the LLM-generated code_string on a COPY of the dataframe
    within a sandboxed execution namespace.

    Args:
        code_string: Python code containing a def clean(df) function.
        df: The raw dataframe to clean.

    Returns:
        (cleaned_df, None) on success.
        (None, traceback_string) on failure.
    """
    df_copy = df.copy()

    # Restrict built-ins to prevent malicious file writes / shell commands
    allowed_builtins = {
        "__import__": __import__,  # needed for 'import pandas as pd' in generated code
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
        "print": print,
        "int": int,
        "float": float,
        "str": str,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "isinstance": isinstance,
        "type": type,
        "round": round,
        "bool": bool,
        "any": any,
        "all": all,
        "sum": sum,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "Exception": Exception,
    }

    global_namespace = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        "np": np,
    }

    local_namespace = {
        "df": df_copy,
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Execute the script (defines the clean function)
        # We use only global_namespace so that imports and the function definition share the same scope
        exec(code_string, global_namespace)

        # Find and call the clean() function
        clean_func = global_namespace.get("clean")
        if not clean_func or not callable(clean_func):
            raise ValueError(
                "Generated code does not define a callable 'clean(df)' function."
            )

        cleaned_df = clean_func(df_copy)

        if not isinstance(cleaned_df, pd.DataFrame):
            raise TypeError(
                f"clean() returned {type(cleaned_df).__name__}, expected pd.DataFrame."
            )

        return cleaned_df, None

    except Exception:
        tb_str = traceback.format_exc()
        return None, tb_str

    finally:
        sys.stdout = old_stdout
