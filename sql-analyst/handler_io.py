"""
Interactive AI SQL Assistant — Chanterelle handler_io.py

Uses Anthropic Claude with tool-use to answer natural-language questions
about a SQLite database. Renders results as rich Chanterelle visualizations.

Required env vars:
    ANTHROPIC_API_KEY  — your Anthropic API key
    DB_URI             — SQLite URI  (default: see below)
    SCHEMA_JSON_PATH   — path to schema.json (default: see below)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import anthropic

from db_tools import get_schema, sql_query, DataStore, build_chart

# ---------------------------------------------------------------------------
# Config (env vars with sensible defaults for the supermarket example)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

DB_URI = os.getenv(
    "DB_URI",
    "file:"
    + os.path.join(
        _HERE, "..", "..", "..", "Repos", "analyst-MCP-server-example",
        "data", "database", "supermarket.db",
    )
    + "?mode=ro",
)
SCHEMA_JSON_PATH = os.getenv(
    "SCHEMA_JSON_PATH",
    os.path.join(
        _HERE, "..", "..", "..", "Repos", "analyst-MCP-server-example",
        "data", "database", "schema.json",
    ),
)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOOL_ROUNDS = 6
MAX_RESULT_ROWS = 200
MAX_HISTORY_TURNS = 40  # messages kept before trimming

# ---------------------------------------------------------------------------
# Anthropic tool definitions (mirrors db_tools.py interface)
# ---------------------------------------------------------------------------
TOOLS: list[dict] = [
    {
        "name": "sql_query",
        "description": (
            "Execute a read-only SQL SELECT query against the database. "
            "Only SELECT statements are allowed (including CTEs ending in SELECT). "
            "Returns columns, rows (list of dicts), rowcount, truncated flag, "
            "and a cache_key that can be used with create_chart or cache_info. "
            "Use parameterized queries with :name placeholders when incorporating user-supplied values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL SELECT statement to execute.",
                },
                "params": {
                    "type": "object",
                    "description": "Optional named parameters for the query (e.g. {\":name\": \"value\"}).",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_schema",
        "description": (
            "Retrieve the full database schema as JSON. Use this if you need to "
            "re-examine the schema during the conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cache_info",
        "description": (
            "Get metadata about a cached query result: column names, row count, "
            "and a sample of the first 5 rows. Use this to remind yourself what "
            "data is in a cache entry before creating a chart."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cache_key": {
                    "type": "string",
                    "description": "The cache key returned by sql_query.",
                },
            },
            "required": ["cache_key"],
        },
    },
    {
        "name": "cache_release",
        "description": (
            "Release a cached query result to free memory. "
            "Call this when you no longer need the data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cache_key": {
                    "type": "string",
                    "description": "The cache key to release.",
                },
            },
            "required": ["cache_key"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Render a chart from cached query data. You specify column names — "
            "the actual data is resolved server-side. NEVER pass raw data values; "
            "always reference columns by name from a cached result.\n\n"
            "Supported chart_type values:\n"
            "- 'bar': Bar chart. Requires x_column and y_column.\n"
            "- 'line': Line chart. Requires x_column and y_column. "
            "Optional group_column to split into multiple series.\n"
            "- 'scatter': Scatter plot. Requires x_column and y_column.\n"
            "- 'histogram': Histogram of a single column. Requires x_column.\n"
            "- 'plotly': Advanced Plotly chart. Provide plotly_spec with trace "
            "templates that use column references (see chart skill below)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cache_key": {
                    "type": "string",
                    "description": "The cache key from a previous sql_query result.",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "histogram", "plotly"],
                    "description": "The type of chart to render.",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title.",
                },
                "x_column": {
                    "type": "string",
                    "description": "Column name for the x-axis / categories.",
                },
                "y_column": {
                    "type": "string",
                    "description": "Column name for the y-axis / values.",
                },
                "group_column": {
                    "type": "string",
                    "description": "Column to group/split into multiple series (line charts).",
                },
                "plotly_spec": {
                    "type": "object",
                    "description": (
                        "For chart_type='plotly' only. A dict with 'traces' (list of trace templates) "
                        "and optional 'layout'. Trace templates use column references like "
                        "{'x_column': 'col_a', 'y_column': 'col_b', 'type': 'bar'}."
                    ),
                },
            },
            "required": ["cache_key", "chart_type", "title"],
        },
    },
]

# ---------------------------------------------------------------------------
# Module-level state (persists across turns — Chanterelle keeps the process alive)
# ---------------------------------------------------------------------------
client: Optional[anthropic.Anthropic] = None
messages: List[Dict[str, Any]] = []
system_prompt: str = ""
schema_cache: Optional[Dict[str, Any]] = None
data_store: Optional[DataStore] = None
_initialized: bool = False


def _build_system_prompt(schema: Dict[str, Any]) -> str:
    """Create the system prompt with the schema and chart skill embedded."""
    schema_text = json.dumps(schema, indent=2)
    return (
        "You are a helpful data analyst assistant. You have access to a read-only "
        "SQLite database. The database schema is provided below.\n\n"
        "## Database Schema\n"
        f"```json\n{schema_text}\n```\n\n"
        "## Guidelines\n"
        "- Write correct SQLite SQL. Use the schema above to determine table and column names.\n"
        "- Always use parameterized queries (:name style) when incorporating literal values from the user.\n"
        "- Only SELECT queries are allowed. Never attempt INSERT, UPDATE, DELETE, DROP, etc.\n"
        "- When results are large, summarize key findings rather than listing every row.\n"
        "- If the user's question is ambiguous, ask for clarification before querying.\n"
        "- You can make multiple queries in one turn if needed to answer the question.\n"
        "- Be concise but thorough in your analysis.\n\n"
        "## Data Cache\n"
        "Every sql_query result is automatically cached. The response includes a `cache_key`.\n"
        "Use `cache_info` to inspect a cached result (columns, row count, sample rows).\n"
        "Use `cache_release` when you no longer need the data.\n\n"
        "## Chart Skill — create_chart tool\n"
        "Use the `create_chart` tool to visualize cached query results. Charts are rendered "
        "directly from cached data — **never pass raw data values in your arguments**. "
        "Only reference column names.\n\n"
        "### Supported chart types\n"
        "| chart_type | Required columns | Optional | Notes |\n"
        "|------------|-----------------|----------|-------|\n"
        "| `bar` | x_column, y_column | — | One bar per row |\n"
        "| `line` | x_column, y_column | group_column | group_column splits into multiple series |\n"
        "| `scatter` | x_column, y_column | — | One point per row |\n"
        "| `histogram` | x_column | — | Distribution of a single column |\n"
        "| `plotly` | — | — | Advanced: provide plotly_spec (see below) |\n\n"
        "### Plotly escape hatch\n"
        "For charts not covered above, use `chart_type='plotly'` with a `plotly_spec` dict:\n"
        "```json\n"
        '{\n'
        '  "traces": [\n'
        '    {"type": "bar", "x_column": "category", "y_column": "total_sales",\n'
        '     "text_column": "label", "name": "Sales"}\n'
        '  ],\n'
        '  "layout": {"title": "Sales by Category", "barmode": "group"}\n'
        '}\n'
        "```\n"
        "Each trace template supports: `x_column`, `y_column`, `text_column`, "
        "`color_column` (resolved to data), plus any standard Plotly trace keys "
        "(`type`, `mode`, `name`, `marker`, etc.).\n\n"
        "### When to create charts\n"
        "- Create charts when they add value — comparisons, trends, distributions.\n"
        "- For simple counts or single values, text is sufficient.\n"
        "- Choose the chart type that best fits the data shape.\n"
        "- Always provide a clear title.\n"
    )


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def _execute_tool(name: str, input_args: Dict[str, Any]) -> str:
    """Run a tool and return the JSON-serialized result string."""
    if name == "sql_query":
        result = sql_query(
            sql=input_args["sql"],
            db_uri=DB_URI,
            params=input_args.get("params"),
            max_rows=MAX_RESULT_ROWS,
            store=data_store,
        )
        return json.dumps(result, default=str)
    elif name == "get_schema":
        result = get_schema(SCHEMA_JSON_PATH)
        return json.dumps(result, default=str)
    elif name == "cache_info":
        if data_store is None:
            return json.dumps({"error": "Data store not initialized"})
        info = data_store.info(input_args["cache_key"])
        if info is None:
            return json.dumps({"error": f"Cache key not found: {input_args['cache_key']}"})
        return json.dumps(info, default=str)
    elif name == "cache_release":
        if data_store is None:
            return json.dumps({"error": "Data store not initialized"})
        ok = data_store.release(input_args["cache_key"])
        return json.dumps({"released": ok, "cache_key": input_args["cache_key"]})
    elif name == "create_chart":
        if data_store is None:
            return json.dumps({"error": "Data store not initialized"})
        result = build_chart(
            store=data_store,
            cache_key=input_args["cache_key"],
            chart_type=input_args["chart_type"],
            title=input_args.get("title", "Chart"),
            x_column=input_args.get("x_column"),
            y_column=input_args.get("y_column"),
            group_column=input_args.get("group_column"),
            plotly_spec=input_args.get("plotly_spec"),
        )
        return json.dumps(result, default=str)
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Output formatting helpers (Chanterelle visualization format)
# ---------------------------------------------------------------------------
def _make_text_section(text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """Wrap markdown text in a Chanterelle section."""
    section: Dict[str, Any] = {
        "type": "section",
        "items": [{"type": "markdown", "content": text}],
    }
    if title:
        section["title"] = title
    return section


def _make_tool_progress_section(
    tool_name: str, tool_input: Dict[str, Any], result_summary: str
) -> Dict[str, Any]:
    """Collapsible section showing what tool was called."""
    if tool_name == "sql_query":
        sql_text = tool_input.get("sql", "")
        content = f"```sql\n{sql_text}\n```\n\n{result_summary}"
        title = "Tool: sql_query"
    elif tool_name == "get_schema":
        content = result_summary
        title = "Tool: get_schema"
    elif tool_name == "create_chart":
        chart_type = tool_input.get("chart_type", "chart")
        chart_title = tool_input.get("title", "")
        content = f"**{chart_type}** — {chart_title}\n\n{result_summary}"
        title = "Tool: create_chart"
    elif tool_name == "cache_info":
        content = result_summary
        title = "Tool: cache_info"
    elif tool_name == "cache_release":
        content = result_summary
        title = "Tool: cache_release"
    else:
        content = result_summary
        title = f"Tool: {tool_name}"

    return {
        "type": "section",
        # "title": title,
        # "collapsible": True,
        # "collapsed": True,
        "items": [{
            "type": "markdown", 
            "title": title,
            "collapsible": True,
            "collapsed": True,
            "content": content
            }],
    }


def _make_table_item(
    columns: List[str], rows: List[Dict[str, Any]], title: str = "Query Results"
) -> Dict[str, Any]:
    return {
        "type": "table",
        "collapsible": True,
        "collapsed": True,
        "title": title,
        "data": {
            "columns": [{"header": c, "field": c} for c in columns],
            "rows": rows,
        },
    }


def _summarize_tool_result(name: str, result_str: str) -> str:
    """Short human-readable summary for the progress section."""
    try:
        data = json.loads(result_str)
    except Exception:
        return "Result could not be parsed."

    if name == "sql_query":
        rc = data.get("rowcount", 0)
        trunc = data.get("truncated", False)
        cols = data.get("columns", [])
        cache_key = data.get("cache_key", "")
        s = f"**{rc} row(s)** returned"
        if trunc:
            s += f" (truncated to {data.get('max_rows', '?')})"
        if cols:
            s += f"  \nColumns: `{'`, `'.join(cols)}`"
        if cache_key:
            s += f"  \nCached as `{cache_key}`"
        return s
    elif name == "get_schema":
        tables = list(
            data.get("properties", {}).get("tables", {}).get("properties", {}).keys()
        )
        return f"Schema loaded — {len(tables)} table(s)"
    elif name == "create_chart":
        if "error" in data:
            return f"**Error:** {data['error']}"
        item_type = data.get("type", "chart")
        return f"Chart rendered ({item_type})"
    elif name == "cache_info":
        if "error" in data:
            return f"**Error:** {data['error']}"
        rc = data.get("rowcount", "?")
        cols = data.get("columns", [])
        return f"**{rc} row(s)** — columns: `{'`, `'.join(cols)}`"
    elif name == "cache_release":
        released = data.get("released", False)
        return "Released" if released else "Key not found"
    return f"Result: {result_str[:200]}"


def _build_final_outputs(
    assistant_text: str, tool_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Combine the assistant's text answer with tables and chart visualizations."""
    outputs: List[Dict[str, Any]] = []

    for tr in tool_results:
        if tr.get("name") == "sql_query":
            try:
                data = json.loads(tr["result_str"])
                cols = data.get("columns", [])
                rows = data.get("rows", [])
                if cols and rows:
                    outputs.append(
                        {"type": "section", "items": [_make_table_item(cols, rows)]}
                    )
            except Exception:
                pass
        elif tr.get("name") == "create_chart":
            try:
                chart_item = json.loads(tr["result_str"])
                if "error" not in chart_item:
                    outputs.append({"type": "section", "items": [chart_item]})
            except Exception:
                pass

    # The assistant's text response
    if assistant_text.strip():
        outputs.append(_make_text_section(assistant_text))

    return outputs


def _trim_history() -> None:
    """Keep conversation history bounded."""
    global messages
    if len(messages) > MAX_HISTORY_TURNS:
        messages = messages[-MAX_HISTORY_TURNS:]


def _chat_input() -> List[Dict[str, Any]]:
    """Standard next_inputs for the chat text box."""
    return [
        {
            "name": "message",
            "label": "Message",
            "type": "textarea",
            "constraints": {"placeholder": "Ask a question about the data..."},
        }
    ]


# ---------------------------------------------------------------------------
# Chanterelle entry points
# ---------------------------------------------------------------------------
def initialize(conversation_history=None):
    """Called when the interactive session starts (or restarts from feedback)."""
    global client, messages, system_prompt, schema_cache, data_store, _initialized

    messages = []
    schema_cache = None
    data_store = None
    _initialized = False

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "outputs": [
                _make_text_section(
                    "**Error:** `ANTHROPIC_API_KEY` environment variable is not set.\n\n"
                    "Please set it before starting this agent."
                )
            ],
            "next_inputs": _chat_input(),
        }

    client = anthropic.Anthropic(api_key=api_key)

    try:
        schema_cache = get_schema(SCHEMA_JSON_PATH)
        system_prompt = _build_system_prompt(schema_cache)
    except Exception as e:
        return {
            "outputs": [
                _make_text_section(
                    f"**Error loading schema:** {e}\n\n"
                    f"Checked path: `{SCHEMA_JSON_PATH}`\n\n"
                    "Set the `SCHEMA_JSON_PATH` environment variable to the correct path."
                )
            ],
            "next_inputs": _chat_input(),
        }

    data_store = DataStore()
    _initialized = True

    # Restore conversation history from a previous feedback session
    if conversation_history:
        messages = conversation_history
        _trim_history()
        return {
            "outputs": [_make_text_section("Session restored from feedback.")],
            "next_inputs": _chat_input(),
        }

    # Welcome message listing the tables
    tables_info = (
        schema_cache.get("properties", {}).get("tables", {}).get("properties", {})
    )
    table_list = []
    for key, tbl in tables_info.items():
        name = tbl.get("table_name", key)
        desc = tbl.get("description", "")
        table_list.append(f"- **{name}**: {desc}")

    welcome_md = (
        "I'm an AI data analyst connected to a SQLite database. "
        "Ask me anything about the data and I'll write queries to find answers.\n\n"
        "### Available Tables\n" + "\n".join(table_list)
    )

    return {
        "outputs": [_make_text_section(welcome_md, title="SQL Assistant")],
        "next_inputs": _chat_input(),
    }


def on_input(data):
    """Called each turn. Generator that yields tool progress then final answer."""
    global messages

    if not _initialized or client is None:
        yield {
            "outputs": [
                _make_text_section("Session not initialized. Please restart the agent.")
            ],
            "next_inputs": _chat_input(),
        }
        return

    user_text = (data.get("message") or "").strip()
    if not user_text:
        yield {
            "outputs": [_make_text_section("Please enter a question or message.")],
            "next_inputs": _chat_input(),
        }
        return

    messages.append({"role": "user", "content": user_text})
    _trim_history()

    # Tool-use loop
    tool_results_for_display: List[Dict[str, Any]] = []

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            yield {
                "outputs": [_make_text_section(f"**API Error:** {e}")],
                "next_inputs": _chat_input(),
            }
            return

        # Process response content blocks
        assistant_text_parts: List[str] = []
        tool_use_blocks: List[Dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                assistant_text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )

        # Record the full assistant message in history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we're done
        if response.stop_reason != "tool_use" or not tool_use_blocks:
            final_text = "\n\n".join(assistant_text_parts)
            outputs = _build_final_outputs(final_text, tool_results_for_display)
            if not outputs:
                outputs = [_make_text_section(final_text or "*(No response)*")]
            yield {"outputs": outputs, "next_inputs": _chat_input()}
            return

        # Execute each tool call, yield progress, build tool_result messages
        tool_result_contents: List[Dict[str, Any]] = []

        for tc in tool_use_blocks:
            try:
                result_str = _execute_tool(tc["name"], tc["input"])
            except Exception as e:
                result_str = json.dumps({"error": str(e)})

            summary = _summarize_tool_result(tc["name"], result_str)
            progress_section = _make_tool_progress_section(
                tc["name"], tc["input"], summary
            )

            # Yield partial progress (no next_inputs — not a prompt)
            yield {"outputs": [progress_section]}

            tool_results_for_display.append(
                {"name": tc["name"], "result_str": result_str}
            )

            # Truncate large results before sending back to Claude
            truncated_result = result_str
            if len(result_str) > 20000:
                truncated_result = result_str[:20000] + "\n...(truncated)"

            tool_result_contents.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": truncated_result,
                }
            )

        # Append tool results to history and continue the loop
        messages.append({"role": "user", "content": tool_result_contents})

    # Exhausted all rounds
    yield {
        "outputs": [
            _make_text_section(
                "I made several tool calls but couldn't produce a final answer. "
                "Please try rephrasing your question."
            )
        ],
        "next_inputs": _chat_input(),
    }
