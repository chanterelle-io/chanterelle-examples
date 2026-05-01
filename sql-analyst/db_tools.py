"""
Standalone read-only SQLite query tools + in-process data cache + chart builder.

Pure functions and a simple DataStore class — no MCP decorators.
Mirrors the safety model of the analyst-MCP-server-example:
  - PRAGMA query_only=ON
  - Only SELECT statements allowed (including CTEs ending in SELECT)
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

MAX_ROWS = 500
MAX_CACHE_ENTRIES = 8

_SELECT_RE = re.compile(
    r"^\s*(with\b.*?select\b|select\b)", re.IGNORECASE | re.DOTALL
)


def _ensure_select(sql: str) -> None:
    if not _SELECT_RE.match(sql or ""):
        raise ValueError(
            "Only SELECT statements (including CTEs ending in SELECT) are allowed."
        )


def get_schema(schema_json_path: str) -> Dict[str, Any]:
    """Read and return the database schema JSON file."""
    with open(schema_json_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data Store (in-process cache keyed by string)
# ---------------------------------------------------------------------------
class DataStore:
    """Simple LRU cache for query results."""

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES):
        self._max = max_entries
        self._data: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def store(self, columns: List[str], rows: List[Dict[str, Any]]) -> str:
        """Store columns+rows, return a new cache key."""
        key = uuid.uuid4().hex[:12]
        self._data[key] = {
            "columns": columns,
            "rows": rows,
            "created_at": time.time(),
        }
        self._touch(key)
        self._evict()
        return key

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._data:
            return None
        self._touch(key)
        return self._data[key]

    def info(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._data.get(key)
        if entry is None:
            return None
        return {
            "cache_key": key,
            "columns": entry["columns"],
            "rowcount": len(entry["rows"]),
            "created_at": entry["created_at"],
        }

    def list_keys(self) -> List[Dict[str, Any]]:
        return [self.info(k) for k in self._order if k in self._data]

    def release(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            if key in self._order:
                self._order.remove(key)
            return True
        return False

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict(self) -> None:
        while len(self._order) > self._max:
            old = self._order.pop(0)
            self._data.pop(old, None)


def sql_query(
    sql: str,
    db_uri: str,
    params: Optional[Dict[str, Any]] = None,
    max_rows: int = MAX_ROWS,
    store: Optional[DataStore] = None,
) -> Dict[str, Any]:
    """Execute a read-only SELECT and return columns + rows.

    If a DataStore is provided, results are auto-cached and cache_key is returned.

    Returns:
        {columns, rows, rowcount, truncated, max_rows, cache_key}
    """
    _ensure_select(sql)

    conn = sqlite3.connect(db_uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON;")
        cur = conn.cursor()
        cur.execute(sql, params or {})
        columns = [d[0] for d in cur.description] if cur.description else []

        rows: List[Dict[str, Any]] = []
        while True:
            batch = cur.fetchmany(500)
            if not batch:
                break
            for r in batch:
                rows.append(dict(zip(columns, r)))
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break

        cache_key = None
        if store is not None:
            cache_key = store.store(columns, rows)

        return {
            "columns": columns,
            "rows": rows,
            "rowcount": len(rows),
            "truncated": len(rows) >= max_rows,
            "max_rows": max_rows,
            "cache_key": cache_key,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chart builder — resolves column references against cached data
# ---------------------------------------------------------------------------
def build_chart(
    store: DataStore,
    cache_key: str,
    chart_type: str,
    title: str = "",
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    group_column: Optional[str] = None,
    plotly_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Chanterelle chart item from cached data and column references.

    Supported chart_type values:
      bar, line, scatter, histogram, plotly

    Returns a Chanterelle visualization item dict (ready to embed in a section).
    """
    entry = store.get(cache_key)
    if entry is None:
        raise ValueError(f"Unknown cache_key: {cache_key}")

    rows = entry["rows"]
    columns = entry["columns"]

    def _col_values(col: str) -> list:
        if col not in columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {columns}"
            )
        return [r.get(col) for r in rows]

    # ----- bar_chart -----
    if chart_type == "bar":
        if not x_column or not y_column:
            raise ValueError("bar chart requires x_column and y_column")
        return {
            "type": "bar_chart",
            "title": title,
            "data": {
                "bars": [
                    {"label": str(x), "value": y}
                    for x, y in zip(_col_values(x_column), _col_values(y_column))
                ],
                "axis": {
                    "x": {"label": x_column},
                    "y": {"label": y_column},
                },
            },
        }

    # ----- line_chart -----
    _LINE_COLORS = [
        "#1976d2", "#e53935", "#43a047", "#fb8c00", "#8e24aa",
        "#00acc1", "#d81b60", "#6d4c41", "#546e7a", "#fdd835",
    ]

    if chart_type == "line":
        if not x_column or not y_column:
            raise ValueError("line chart requires x_column and y_column")
        xs = _col_values(x_column)
        if group_column:
            groups: Dict[str, list] = {}
            for x, y, g in zip(xs, _col_values(y_column), _col_values(group_column)):
                groups.setdefault(str(g), []).append({"x": x, "y": y})
            lines = [
                {
                    "id": gname,
                    "points": pts,
                    "style": {"color": _LINE_COLORS[i % len(_LINE_COLORS)]},
                }
                for i, (gname, pts) in enumerate(groups.items())
            ]
        else:
            lines = [
                {
                    "id": y_column,
                    "points": [
                        {"x": x, "y": y}
                        for x, y in zip(xs, _col_values(y_column))
                    ],
                }
            ]
        return {
            "type": "line_chart",
            "title": title,
            "data": {
                "lines": lines,
                "axis": {
                    "x": {"label": x_column},
                    "y": {"label": y_column},
                },
            },
        }

    # ----- scatter_plot -----
    if chart_type == "scatter":
        if not x_column or not y_column:
            raise ValueError("scatter chart requires x_column and y_column")
        points = [
            {"x": x, "y": y}
            for x, y in zip(_col_values(x_column), _col_values(y_column))
        ]
        item: Dict[str, Any] = {
            "type": "scatter_plot",
            "title": title,
            "data": {
                "points": points,
                "axis": {
                    "x": {"label": x_column},
                    "y": {"label": y_column},
                },
            },
        }
        return item

    # ----- histogram (via plotly) -----
    if chart_type == "histogram":
        if not x_column:
            raise ValueError("histogram requires x_column")
        return {
            "type": "plotly",
            "title": title,
            "data": [{"x": _col_values(x_column), "type": "histogram"}],
            "layout": {"title": title, "xaxis": {"title": x_column}},
        }

    # ----- plotly (advanced — trace templates with column references) -----
    if chart_type == "plotly":
        if not plotly_spec:
            raise ValueError("plotly chart_type requires plotly_spec")
        traces = []
        for tmpl in plotly_spec.get("traces", []):
            trace: Dict[str, Any] = {}
            for k, v in tmpl.items():
                if k == "x_column":
                    trace["x"] = _col_values(v)
                elif k == "y_column":
                    trace["y"] = _col_values(v)
                elif k == "text_column":
                    trace["text"] = _col_values(v)
                elif k == "color_column":
                    trace["marker"] = {"color": _col_values(v)}
                else:
                    trace[k] = v
            traces.append(trace)
        layout = plotly_spec.get("layout", {})
        if title and "title" not in layout:
            layout["title"] = title
        return {
            "type": "plotly",
            "title": title,
            "data": traces,
            "layout": layout,
        }

    raise ValueError(
        f"Unknown chart_type: '{chart_type}'. "
        "Supported: bar, line, scatter, histogram, plotly"
    )
