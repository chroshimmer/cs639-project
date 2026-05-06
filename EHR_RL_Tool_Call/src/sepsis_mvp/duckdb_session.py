#!/usr/bin/env python3
"""
DuckDB Code Execution Session
=============================

Provides a safe, sandboxed Python execution environment backed by DuckDB
for running LLM-generated code that queries clinical databases.

Features:
- Read-only DuckDB connection
- Code safety validation (blocks dangerous operations)
- Execution timeout to prevent infinite loops
- Query row/memory limits to prevent OOM
- Output truncation

Usage:
    session = DuckDBCodeSession(db_path="/path/to/database.db")
    result = session.execute("print(query_db('SELECT * FROM patients LIMIT 5'))")
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
import threading
import traceback
import types
import inspect
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = "/data/ossowski/MIMIC-IV/3.1/mimic4_duck_derived.db"
MAX_OUTPUT_LENGTH = 8192

SAFE_EXEC_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "next": next,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}

# ---------------------------------------------------------------------------
# Time-bounded view templates
# ---------------------------------------------------------------------------
# Used by DuckDBCodeSession when time_bounded_views=True.
# Each entry maps (schema, table) -> a SELECT body (no leading "SELECT",
# no "CREATE VIEW" prefix).  Two placeholders are substituted at runtime:
#   {stay_id}    - integer ICU stay identifier
#   {hadm_id}    - integer hospital admission identifier
#   {visible_ts} - TIMESTAMP literal, e.g. TIMESTAMP '2150-06-01 08:00:00'
# Tables not listed here receive an unfiltered passthrough view so that
# lookup/reference tables and any extra schemas stay accessible.
_TIME_BOUNDED_VIEW_TEMPLATES: Dict[tuple, str] = {
    # ---- ICU tables -------------------------------------------------------
    ("mimiciv_icu", "icustays"): (
        "SELECT stay_id, subject_id, hadm_id, intime,\n"
        "       CASE WHEN outtime <= {visible_ts} THEN outtime ELSE {visible_ts} END AS outtime,\n"
        "       first_careunit, last_careunit, los\n"
        "FROM source.mimiciv_icu.icustays\n"
        "WHERE stay_id = {stay_id}"
    ),
    ("mimiciv_icu", "chartevents"): (
        "SELECT * FROM source.mimiciv_icu.chartevents\n"
        "WHERE stay_id = {stay_id} AND charttime <= {visible_ts}"
    ),
    ("mimiciv_icu", "inputevents"): (
        "SELECT * FROM source.mimiciv_icu.inputevents\n"
        "WHERE stay_id = {stay_id} AND starttime <= {visible_ts}"
    ),
    ("mimiciv_icu", "outputevents"): (
        "SELECT * FROM source.mimiciv_icu.outputevents\n"
        "WHERE stay_id = {stay_id} AND charttime <= {visible_ts}"
    ),
    ("mimiciv_icu", "procedureevents"): (
        "SELECT * FROM source.mimiciv_icu.procedureevents\n"
        "WHERE stay_id = {stay_id} AND starttime <= {visible_ts}"
    ),
    # ---- Hospital tables --------------------------------------------------
    ("mimiciv_hosp", "admissions"): (
        "SELECT * FROM source.mimiciv_hosp.admissions\n"
        "WHERE hadm_id = {hadm_id}"
    ),
    ("mimiciv_hosp", "diagnoses_icd"): (
        "SELECT * FROM source.mimiciv_hosp.diagnoses_icd\n"
        "WHERE hadm_id = {hadm_id}"
    ),
    ("mimiciv_hosp", "procedures_icd"): (
        "SELECT * FROM source.mimiciv_hosp.procedures_icd\n"
        "WHERE hadm_id = {hadm_id}"
    ),
    ("mimiciv_hosp", "labevents"): (
        "SELECT * FROM source.mimiciv_hosp.labevents\n"
        "WHERE hadm_id = {hadm_id} AND charttime <= {visible_ts}"
    ),
    ("mimiciv_hosp", "microbiologyevents"): (
        "SELECT * FROM source.mimiciv_hosp.microbiologyevents\n"
        "WHERE hadm_id = {hadm_id}\n"
        "  AND COALESCE(charttime, CAST(chartdate AS TIMESTAMP)) <= {visible_ts}"
    ),
    ("mimiciv_hosp", "prescriptions"): (
        "SELECT * FROM source.mimiciv_hosp.prescriptions\n"
        "WHERE hadm_id = {hadm_id} AND starttime <= {visible_ts}"
    ),
    ("mimiciv_hosp", "pharmacy"): (
        "SELECT * FROM source.mimiciv_hosp.pharmacy\n"
        "WHERE hadm_id = {hadm_id} AND starttime <= {visible_ts}"
    ),
    ("mimiciv_hosp", "poe"): (
        "SELECT * FROM source.mimiciv_hosp.poe\n"
        "WHERE hadm_id = {hadm_id} AND ordertime <= {visible_ts}"
    ),
    ("mimiciv_hosp", "emar"): (
        "SELECT * FROM source.mimiciv_hosp.emar\n"
        "WHERE hadm_id = {hadm_id} AND charttime <= {visible_ts}"
    ),
}

# Safety limits to prevent system crashes
CODE_EXECUTION_TIMEOUT = 60  # seconds - max time for any code block to run
SQL_QUERY_TIMEOUT = 120  # seconds - max time for any SQL query
MAX_QUERY_ROWS = 100000  # max rows returned by query_db
MAX_QUERY_MEMORY_MB = 500  # approx max memory for query results

# Dangerous patterns to block in LLM-generated code
BLOCKED_IMPORTS = frozenset({
    'subprocess', 'os.system', 'shutil', 'socket',
    'multiprocessing', 'ctypes', 'pty', 'fcntl',
})
BLOCKED_FUNCTIONS = frozenset({
    'exec', 'eval', 'compile', 'open', '__import__',
    'getattr', 'setattr', 'delattr', 'globals', 'locals',
    'exit', 'quit', 'input',
})
# Block direct database connection calls - LLM must use query_db()
BLOCKED_METHODS = frozenset({
    'connect',  # Block duckdb.connect() - must use query_db()
})
BLOCKED_ATTRIBUTES = frozenset({
    'system', 'popen', 'spawn', 'fork', 'kill',
    'remove', 'rmdir', 'unlink', 'rmtree',
    'shutdown', 'reboot',
})


# =========================================================================
# Exceptions
# =========================================================================

class CodeExecutionTimeout(Exception):
    """Raised when code execution exceeds the allowed time."""
    pass


class UnsafeCodeError(Exception):
    """Raised when code contains potentially dangerous operations."""
    pass


# =========================================================================
# Code safety validation
# =========================================================================

def _validate_code_safety(code: str) -> None:
    """
    Check code for dangerous patterns before execution.
    Raises UnsafeCodeError if unsafe patterns are detected.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let execution handle syntax errors
        return
    
    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split('.')[0]
                if name in BLOCKED_IMPORTS or alias.name in BLOCKED_IMPORTS:
                    raise UnsafeCodeError(
                        f"Import '{alias.name}' is not allowed for security reasons. "
                        f"Use query_db() to access the database."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module.split('.')[0] in BLOCKED_IMPORTS or module in BLOCKED_IMPORTS:
                raise UnsafeCodeError(
                    f"Import from '{module}' is not allowed for security reasons."
                )
        
        # Block dangerous function calls
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in BLOCKED_FUNCTIONS:
                    raise UnsafeCodeError(
                        f"Function '{func.id}' is not allowed for security reasons."
                    )
            elif isinstance(func, ast.Attribute):
                if func.attr in BLOCKED_ATTRIBUTES:
                    raise UnsafeCodeError(
                        f"Method '{func.attr}' is not allowed for security reasons."
                    )
                # Block duckdb.connect() - must use query_db()
                if isinstance(func.value, ast.Name) and func.value.id == 'duckdb':
                    if func.attr in BLOCKED_METHODS:
                        raise UnsafeCodeError(
                            f"duckdb.{func.attr}() is not allowed. "
                            f"Use query_db(sql) to execute database queries instead."
                        )
                # Block os.* dangerous calls
                if isinstance(func.value, ast.Name) and func.value.id == 'os':
                    if func.attr in {'system', 'popen', 'spawn', 'fork', 'kill',
                                    'remove', 'rmdir', 'unlink', 'execv', 'execve',
                                    '_exit'}:
                        raise UnsafeCodeError(
                            f"os.{func.attr}() is not allowed for security reasons."
                        )


def _run_with_timeout(func, timeout_sec: int, *args, **kwargs):
    """
    Run a function with a timeout. Returns (result, error).
    Uses threading to avoid signal issues with non-main threads.
    """
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_sec)
    
    if thread.is_alive():
        # Thread is still running - it will be abandoned (daemon thread)
        raise CodeExecutionTimeout(
            f"Code execution exceeded {timeout_sec}s timeout. "
            f"This may indicate an infinite loop or very slow query. "
            f"Please optimize your code."
        )
    
    if error[0] is not None:
        raise error[0]
    
    return result[0]


# =========================================================================
# Code block extraction
# =========================================================================

def extract_code_blocks(text: str) -> List[str]:
    """
    Extract executable code blocks from LLM output.

    Looks for (in priority order):
      1. <code>…</code> XML tags  (including unclosed trailing <code>)
      2. ```python … ``` markdown fences  (only if no <code> tag at all)
      3. ``` … ``` generic fences         (only if no <code> tag at all)

    Returns a list of code strings (may be empty).
    """
    # --- Pre-processing: strip <think>…</think> reasoning blocks ---
    # Some models (e.g. Qwen) emit <think>…</think> around chain-of-thought.
    # Remove them so they don't interfere with code extraction.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also strip orphaned </think> tags (model sometimes omits the opening tag)
    cleaned = cleaned.replace("</think>", "")
    # Normalize <code ...> variants (e.g. Gemma emits "<code >") to plain <code>
    cleaned = re.sub(r"<code[^>]*>", "<code>", cleaned)

    # 1a. Fully-closed <code>…</code> tags
    blocks = re.findall(r"<code>(.*?)</code>", cleaned, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]

    # 1b. Unclosed <code> tag (LLM response truncated by max_tokens).
    #     Extract from the last <code> to end-of-text so we run the
    #     actual (partial) code instead of falling through to markdown
    #     fences that may contain non-code content like data tables.
    if "<code>" in cleaned:
        last_open = cleaned.rfind("<code>")
        trailing = cleaned[last_open + len("<code>"):].strip()
        if trailing:
            return [trailing]

    # --- Fallback: markdown fences (only when NO <code> tag present) ---
    # 2. ```python fences
    blocks = re.findall(r"```python\s*\n(.*?)```", cleaned, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # 3. generic fences
    blocks = re.findall(r"```\s*\n(.*?)```", cleaned, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    return []


# =========================================================================
# DuckDB Code Execution Session
# =========================================================================

class DuckDBCodeSession:
    """
    Persistent code-execution namespace backed by DuckDB.

    Pre-loads a read-only DuckDB connection and helper utilities so
    the agent can explore the schema and run queries immediately.
    
    Safety features:
    - Code validation to block dangerous operations
    - Execution timeout to prevent infinite loops  
    - Query row/memory limits
    - Output truncation
    """

    def __init__(
        self,
        db_path: str,
        subject_id: Optional[int] = None,
        hadm_id: Optional[int] = None,
        cutoff_date: Optional[str] = None,
        max_output_len: int = MAX_OUTPUT_LENGTH,
        guidelines_dir: Optional[str] = None,
        functions_dir: Optional[str] = None,
        # Time-bounded views mode (for rolling-monitor use)
        stay_id: Optional[int] = None,
        visible_until: Optional[Any] = None,
        time_bounded_views: bool = False,
        allowed_relations: Optional[set[tuple[str, str]]] = None,
    ):
        """Create a DuckDB-backed code execution session.

        Parameters
        ----------
        db_path:
            Path to the DuckDB database file.
        subject_id, hadm_id, cutoff_date:
            Optional patient identifiers / cutoff seeded into the namespace
            as ``SUBJECT_ID``, ``HADM_ID``, ``_CUTOFF_DATE``.  Used by the
            autoformalize pipeline; ignored when *time_bounded_views* is True
            (those values are derived from *stay_id* / *visible_until*).
        stay_id:
            ICU stay identifier.  Required when *time_bounded_views=True*.
            Also seeded into the namespace as ``stay_id`` (lower-case) when
            provided.
        visible_until:
            ``datetime`` (or ISO string) representing the exclusive time
            horizon for the current checkpoint.  Required when
            *time_bounded_views=True*.
        time_bounded_views:
            When *True*, create an in-memory DuckDB connection that ATTACHes
            the source database read-only and exposes every table through a
            view.  Tables in ``_TIME_BOUNDED_VIEW_TEMPLATES`` are filtered to
            only return rows visible at *visible_until*; all other tables
            receive an unfiltered passthrough view so that lookup tables and
            extra schemas (e.g. ``mimiciv_derived``) remain accessible.
            When *False* (default), open *db_path* directly as read-only —
            the original behaviour, required for backward compatibility with
            the autoformalize pipeline.
        """
        import duckdb as _duckdb

        self.db_path = db_path
        self.max_output_len = max_output_len
        self.namespace: Dict[str, Any] = {}
        self._guidelines_dir = guidelines_dir
        self._functions_dir = functions_dir
        self._allowed_relations = allowed_relations

        # Query timing stats
        self.query_count: int = 0
        self.query_time_total: float = 0.0  # seconds

        # Build the connection ------------------------------------------------
        if time_bounded_views:
            if stay_id is None or visible_until is None or hadm_id is None:
                raise ValueError(
                    "time_bounded_views=True requires stay_id, hadm_id, and visible_until."
                )
            self._connection = self._build_filtered_connection_for(
                _duckdb, stay_id, hadm_id, visible_until
            )
        else:
            # Original behaviour: open the on-disk DB directly as read-only.
            self._connection = _duckdb.connect(db_path, read_only=True)

        # ---- Build query_db as a proper Python closure ----
        # The connection lives *outside* the exec'd namespace so agent
        # code cannot bypass query_db() or access the connection directly.
        _conn = self._connection
        _max_rows = MAX_QUERY_ROWS
        _max_mem = MAX_QUERY_MEMORY_MB

        import pandas as _pd

        def query_db(
            sql: str,
            params: "list | tuple | None" = None,
            limit: "int | None" = None,
        ) -> _pd.DataFrame:
            """Execute a SQL query against the DuckDB database and return
            a pandas DataFrame.

            Results are limited to {max_rows} rows by default.  Use the
            ``limit`` parameter or add ``LIMIT`` to your SQL for smaller
            results.

            NOTE: Do NOT call ``duckdb.connect()`` — use this function.
            """.format(max_rows=_max_rows)

            # --- row limit ---
            effective_limit = limit if limit is not None else _max_rows
            if "LIMIT" not in sql.upper():
                wrapped = f"SELECT * FROM ({sql}) _limited LIMIT {effective_limit}"
            else:
                wrapped = sql

            # --- execute ---
            import time as _time
            _t0 = _time.monotonic()
            if params:
                df = _conn.execute(wrapped, params).fetchdf()
            else:
                df = _conn.execute(wrapped).fetchdf()
            _elapsed = _time.monotonic() - _t0
            self.query_count += 1
            self.query_time_total += _elapsed

            # --- memory guard ---
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if mem_mb > _max_mem:
                raise MemoryError(
                    f"Query result uses ~{mem_mb:.0f}MB which exceeds the "
                    f"{_max_mem}MB limit. Add LIMIT or filters to reduce size."
                )

            return df

        # ---- Seed the exec namespace with safe helpers only ----
        # NOTE: the raw DB connection is intentionally NOT placed in the
        # namespace — all database access must go through query_db().
        self.namespace["query_db"] = query_db

        # ---- Guideline retrieval functions ----
        _gdir = self._guidelines_dir

        def search_guidelines(keyword: str = "") -> list:
            """Search available clinical guidelines by name keyword.

            Parameters
            ----------
            keyword : str
                Filter to guideline names containing this string
                (case-insensitive). If empty, returns a preview of the
                first 20 names and the total count.

            Returns
            -------
            list
                Matching guideline names. Call ``get_guideline(name)`` to
                read the full text of any entry.

            Examples
            --------
            search_guidelines()            # preview all (first 20 + count)
            search_guidelines("sofa")      # names containing "sofa"
            search_guidelines("infect")    # names containing "infect"
            """
            import os as _os
            if _gdir is None or not _os.path.isdir(_gdir):
                return []
            all_names = sorted(
                f[:-4] for f in _os.listdir(_gdir) if f.endswith(".txt")
            )
            if not keyword:
                preview = all_names[:20]
                if len(all_names) > 20:
                    preview.append(
                        f"... ({len(all_names) - 20} more; "
                        "pass a keyword to filter)"
                    )
                return preview
            kw = keyword.lower()
            return [n for n in all_names if kw in n.lower()]

        def get_guideline(name: str) -> str:
            """Retrieve the full text of a clinical guideline by concept name.

            Parameters
            ----------
            name : str
                The concept name (e.g. ``"qsofa"``, ``"sirs"``).

            Returns
            -------
            str
                The guideline text describing the clinical concept,
                its definition, scoring criteria, and clinical context.
            """
            import os as _os
            if _gdir is None:
                return "(no guidelines directory configured)"
            path = _os.path.join(_gdir, name + ".txt")
            if not _os.path.isfile(path):
                return f"(no guideline found for '{name}')"
            with open(path) as _f:
                return _f.read().strip()

        self.namespace["search_guidelines"] = search_guidelines
        self.namespace["get_guideline"] = get_guideline

        # ---- Function lookup from previously completed concepts ----
        _fdir = self._functions_dir

        def search_functions(keyword: str = "") -> list:
            """Search available saved concept functions by name keyword.

            Parameters
            ----------
            keyword : str
                Filter to function names containing this string
                (case-insensitive). If empty, returns a preview of the
                first 20 names and the total count.

            Returns
            -------
            list
                Matching function names. Call ``get_function_info(name)``
                to read the signature/docstring, or ``load_function(name)``
                to import it into the session.

            Examples
            --------
            search_functions()          # preview all (first 20 + count)
            search_functions("sofa")    # names containing "sofa"
            search_functions("vital")   # names containing "vital"
            """
            import os as _os
            if _fdir is None or not _os.path.isdir(_fdir):
                return []
            all_names = sorted(
                f[:-3] for f in _os.listdir(_fdir) if f.endswith(".py")
            )
            if not keyword:
                preview = all_names[:20]
                if len(all_names) > 20:
                    preview.append(
                        f"... ({len(all_names) - 20} more; "
                        "pass a keyword to filter)"
                    )
                return preview
            kw = keyword.lower()
            return [n for n in all_names if kw in n.lower()]

        def get_function_info(name: str) -> str:
            """Return the signature and docstring of a saved concept function.

            Only the function signature and docstring are returned (not
            the full source code) to keep context compact.

            Parameters
            ----------
            name : str
                The concept name (e.g. ``"qsofa"``).

            Returns
            -------
            str
                A summary showing the function's ``def`` line and its
                docstring.
            """
            import os as _os, ast as _ast, textwrap as _tw
            if _fdir is None:
                return "(no functions directory configured)"
            path = _os.path.join(_fdir, name + ".py")
            if not _os.path.isfile(path):
                return f"(no saved function found for '{name}')"
            with open(path) as _f:
                source = _f.read()
            try:
                tree = _ast.parse(source)
            except SyntaxError:
                return "(saved function has syntax errors)"
            parts = []
            for node in _ast.walk(tree):
                if isinstance(node, _ast.FunctionDef):
                    # Build signature line
                    args = _ast.get_source_segment(source, node.args)
                    if args is None:
                        # Fallback: reconstruct from AST
                        arg_names = [a.arg for a in node.args.args]
                        args = ", ".join(arg_names)
                    sig = f"def {node.name}({args}):"
                    # Extract docstring
                    docstring = _ast.get_docstring(node)
                    if docstring:
                        indented = _tw.indent(docstring.strip(), "    ")
                        parts.append(f'{sig}\n    """\n{indented}\n    """')
                    else:
                        parts.append(sig)
            if not parts:
                return "(no function definitions found in saved file)"
            return "\n\n".join(parts)

        def load_function(name: str) -> str:
            """Execute a previously saved concept function into the session.

            After calling this, the function (and any helpers defined in
            the same file) will be available to call directly by name.

            Parameters
            ----------
            name : str
                The concept name (e.g. ``"qsofa"``).

            Returns
            -------
            str
                A confirmation message, or an error description.
            """
            import os as _os
            if _fdir is None:
                return "(no functions directory configured)"
            path = _os.path.join(_fdir, name + ".py")
            if not _os.path.isfile(path):
                return f"(no saved function found for '{name}')"
            with open(path) as _f:
                code = _f.read()
            try:
                exec(code, self.namespace)
                return f"Functions from '{name}' loaded successfully."
            except Exception as _e:
                return f"Error loading functions from '{name}': {_e}"

        self.namespace["search_functions"] = search_functions
        self.namespace["get_function_info"] = get_function_info
        self.namespace["load_function"] = load_function

        # Derive namespace values from time_bounded_views params when active
        _ns_subject_id = subject_id
        _ns_hadm_id = hadm_id
        _ns_cutoff_date = cutoff_date
        _ns_stay_id = stay_id
        _ns_visible_until = visible_until
        if time_bounded_views:
            _ns_hadm_id = hadm_id
            if cutoff_date is None and visible_until is not None:
                _ns_cutoff_date = (
                    visible_until.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(visible_until, "strftime")
                    else str(visible_until)
                )

        init_code = textwrap.dedent(f"""\
            import pandas as pd
            import numpy as np
            import duckdb
            from datetime import datetime, timedelta
            from typing import Optional, List, Tuple, Dict, Any

            DB_PATH = "{db_path}"
            SUBJECT_ID = {_ns_subject_id if _ns_subject_id is not None else 'None'}
            HADM_ID = {_ns_hadm_id if _ns_hadm_id is not None else 'None'}
            STAY_ID = {_ns_stay_id if _ns_stay_id is not None else 'None'}
            _CUTOFF_DATE = {f'"{_ns_cutoff_date}"' if _ns_cutoff_date else 'None'}
        """)
        exec(init_code, self.namespace)
        self.namespace["__builtins__"] = SAFE_EXEC_BUILTINS

        # Inject lower-case aliases used by rolling-monitor prompts
        if _ns_stay_id is not None:
            self.namespace["stay_id"] = _ns_stay_id
        if _ns_subject_id is not None:
            self.namespace["subject_id"] = _ns_subject_id
        if _ns_hadm_id is not None:
            self.namespace["hadm_id"] = _ns_hadm_id
        if _ns_visible_until is not None:
            self.namespace["visible_until"] = _ns_visible_until

    # -----------------------------------------------------------------------
    # Time-bounded connection builder
    # -----------------------------------------------------------------------

    def _build_filtered_connection_for(
        self, duckdb_mod: Any, stay_id: int, hadm_id: int, visible_until: Any
    ) -> Any:
        """Instance method that has access to self.db_path."""
        if hasattr(visible_until, "strftime"):
            ts_str = visible_until.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(visible_until)[:19]
        visible_ts = f"TIMESTAMP '{ts_str}'"
        tpl_ctx = {
            "stay_id": int(stay_id),
            "hadm_id": int(hadm_id),
            "visible_ts": visible_ts,
        }

        conn = duckdb_mod.connect()  # in-memory DuckDB

        safe_path = self.db_path.replace("'", "''")
        conn.execute(f"ATTACH '{safe_path}' AS source (READ_ONLY)")

        # Discover all schemas and tables in the attached source.
        # In DuckDB, information_schema.tables on the in-memory connection
        # covers all attached databases; filter by table_catalog = 'source'.
        rows = conn.execute(
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_catalog = 'source' "
            "  AND table_schema NOT IN ('information_schema', 'pg_catalog')"
        ).fetchall()
        if self._allowed_relations is not None:
            rows = [
                (schema, table)
                for schema, table in rows
                if (schema, table) in self._allowed_relations
            ]

        # Group by schema so we CREATE SCHEMA once per schema
        from collections import defaultdict as _dd
        schema_tables: Dict[str, List[str]] = _dd(list)
        for schema, table in rows:
            schema_tables[schema].append(table)

        for schema, tables in schema_tables.items():
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            for table in tables:
                key = (schema, table)
                if key in _TIME_BOUNDED_VIEW_TEMPLATES:
                    select_body = _TIME_BOUNDED_VIEW_TEMPLATES[key].format(**tpl_ctx)
                else:
                    # Passthrough: lookup tables, derived tables, extra schemas
                    select_body = f"SELECT * FROM source.{schema}.{table}"
                conn.execute(
                    f'CREATE VIEW "{schema}"."{table}" AS {select_body}'
                )

        return conn

    def get_query_stats(self) -> Dict[str, Any]:
        """Return cumulative SQL query stats."""
        return {
            "query_count": self.query_count,
            "query_time_total_sec": round(self.query_time_total, 3),
        }

    def close(self):
        """Close the database connection."""
        conn = getattr(self, "_connection", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._connection = None

    def __del__(self):
        """Ensure connection is closed when session is garbage collected."""
        self.close()

    # ---- helpers --------------------------------------------------------

    def get_function_signatures(self) -> str:
        """Return a formatted list of user-defined functions in the namespace."""
        lines: list[str] = []
        for name, obj in sorted(self.namespace.items()):
            if isinstance(obj, types.FunctionType) and not name.startswith("_"):
                try:
                    sig = inspect.signature(obj)
                    doc = (obj.__doc__ or "").strip().split("\n")[0]
                    lines.append(
                        f"  {name}{sig}  # {doc}" if doc else f"  {name}{sig}"
                    )
                except (ValueError, TypeError):
                    lines.append(f"  {name}(...)")
        return "\n".join(lines) if lines else "(no user functions found)"

    @staticmethod
    def _strip_markdown_fences(code: str) -> str:
        code = code.strip()
        m = re.match(r"^```[\w+-]*\s*\n(.*?)\n?```$", code, re.DOTALL)
        if m:
            return m.group(1).strip()
        if code.startswith("```"):
            first_nl = code.index("\n") if "\n" in code else 3
            code = code[first_nl + 1 :]
            if code.endswith("```"):
                code = code[:-3]
            return code.strip()
        return code

    def execute(self, code: str, timeout: int = CODE_EXECUTION_TIMEOUT) -> str:
        """Execute *code* in the persistent namespace; return output or error.
        
        Includes safety features:
        - Code validation to block dangerous operations
        - Execution timeout to prevent infinite loops
        - Output truncation
        """
        code = self._strip_markdown_fences(code.strip())
        if not code:
            return "(empty code)"

        # Validate code for dangerous patterns BEFORE execution
        try:
            _validate_code_safety(code)
        except UnsafeCodeError as e:
            return f"BLOCKED: {e}"

        stdout_buf = StringIO()
        stderr_buf = StringIO()

        def _do_execute():
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                tree = ast.parse(code)
                if not tree.body:
                    return None

                last_is_expr = isinstance(tree.body[-1], ast.Expr)

                if last_is_expr and len(tree.body) == 1:
                    result = eval(code, self.namespace)
                    if result is not None:
                        if hasattr(result, "to_string"):
                            print(result.to_string())
                        else:
                            print(repr(result))
                elif last_is_expr:
                    module = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(
                        compile(module, "<string>", "exec"), self.namespace
                    )
                    last_expr = ast.Expression(body=tree.body[-1].value)
                    result = eval(
                        compile(last_expr, "<string>", "eval"),
                        self.namespace,
                    )
                    if result is not None:
                        if hasattr(result, "to_string"):
                            print(result.to_string())
                        else:
                            print(repr(result))
                else:
                    exec(code, self.namespace)

        try:
            _run_with_timeout(_do_execute, timeout)

            output = stdout_buf.getvalue()
            err = stderr_buf.getvalue()
            if err:
                output += f"\nStderr:\n{err}"
            if not output.strip():
                output = "Code executed successfully (no output)."
            if len(output) > self.max_output_len:
                output = (
                    output[: self.max_output_len]
                    + f"\n... (truncated, {len(output) - self.max_output_len} more chars)"
                )
            return output
        except CodeExecutionTimeout as e:
            return f"TIMEOUT: {e}"
        except Exception:
            tb = traceback.format_exc()
            if len(tb) > self.max_output_len:
                tb = tb[: self.max_output_len] + "\n... (truncated)"
            return f"Error:\n{tb}"
