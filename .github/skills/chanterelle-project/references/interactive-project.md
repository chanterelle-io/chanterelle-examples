# Interactive Project Reference

Interactive projects enable multi-turn conversational agents with dynamic forms and rich output.

## interactive.json Schema

```json
{
  "interactive_name": "Agent Display Name",
  "version": "1.0.0",
  "description": "Detailed description",
  "description_short": "Brief for catalog card",
  "tags": {"type": "demo", "domain": "nlp"},
  "python_environment": {"type": "system"}
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `interactive_name` | Yes | Name in catalog and page header |
| `version` | No | Version string |
| `description` | No | Detailed description |
| `description_short` | No | Short text for catalog card |
| `tags` | No | Key-value categorization |
| `python_environment` | No | Python env config: `system`, `venv`, `conda`, `virtualenv` |

## handler_io.py

Interactive handlers implement 2 functions:

```python
def initialize(conversation_history=None):
    """Called when session starts (or restarts from feedback).
    
    Args:
        conversation_history: Optional list of {"role": "user"|"assistant", "content": "..."}
                              dicts from a previous session. Passed when the user
                              resumes from a feedback entry. Omit the parameter
                              entirely if you don't need conversation restore.
    """
    return {
        "outputs": [...],       # Sections to display (same as model_findings content)
        "next_inputs": [...]    # Input definitions for the next form
    }

def on_input(data):
    """Called each turn. data = dict of user's input values."""
    return {
        "outputs": [...],       # Sections to display
        "next_inputs": [...]    # Input definitions for the next form
    }
```

The `conversation_history` parameter is **opt-in**: if your `initialize()` doesn't declare it, Chanterelle calls `initialize()` with no arguments as usual. When declared, it receives a list of simplified role/content message dicts extracted from the UI conversation, e.g.:

```python
[
    {"role": "user", "content": "show me top 5 products"},
    {"role": "assistant", "content": "Here are the top 5 products by revenue..."},
    {"role": "user", "content": "now show that as a bar chart"},
    {"role": "assistant", "content": "Here's the bar chart of the top 5 products..."}
]
```

### Response Object

| Field | Type | Description |
|-------|------|-------------|
| `outputs` | array | Sections/items — same format as model_findings.json `content`. See [visualization-types.md](./visualization-types.md) |
| `next_inputs` | array | Input definitions — same format as model_meta.json `inputs` (name, label, type, constraints, default) |

### Composer Auto-Detection

Chanterelle picks the input UI automatically:

| next_inputs pattern | UI shown |
|---------------------|----------|
| Single `string` or `textarea` | Chat-style text box |
| Single `button` or `yes_no` | Inline quick-choice buttons |
| Multiple inputs or complex types | Full multi-field form panel |

### State Management

The Python process persists across turns. Use module-level variables for state:

```python
history = []
user_name = None

def initialize(conversation_history=None):
    global history, user_name
    history = []
    user_name = None
    if conversation_history:
        history = conversation_history  # Restore from feedback session
    return {"outputs": [...], "next_inputs": [...]}

def on_input(data):
    global user_name
    history.append({"role": "user", "content": data.get("message", "")})
    if "username" in data:
        user_name = data["username"]
    # Use history and user_name for context-aware responses
    return {"outputs": [...], "next_inputs": [...]}
```

## Complete Example: Data Explorer Agent

```python
import json

user_data = None

def initialize():
    return {
        "outputs": [{
            "type": "section",
            "title": "Data Explorer",
            "items": [{
                "type": "text",
                "id": "welcome",
                "content": [{"type": "paragraph", "text": "Upload a CSV to explore your data."}]
            }]
        }],
        "next_inputs": [{
            "name": "dataset",
            "label": "Upload CSV",
            "type": "file",
            "constraints": {"extensions": [".csv"]}
        }]
    }

def on_input(data):
    global user_data
    import pandas as pd

    if "dataset" in data:
        user_data = pd.read_csv(data["dataset"])
        cols = list(user_data.columns)
        return {
            "outputs": [{
                "type": "section",
                "title": "Dataset Loaded",
                "items": [
                    {
                        "type": "table",
                        "title": f"Preview ({len(user_data)} rows)",
                        "data": {
                            "columns": [{"header": c, "field": c} for c in cols],
                            "rows": user_data.head(10).to_dict("records")
                        }
                    }
                ]
            }],
            "next_inputs": [
                {
                    "name": "column",
                    "label": "Column to Analyze",
                    "type": "category",
                    "constraints": {"options": cols}
                }
            ]
        }

    if "column" in data and user_data is not None:
        col = data["column"]
        stats = user_data[col].describe()
        return {
            "outputs": [{
                "type": "section",
                "title": f"Analysis: {col}",
                "items": [
                    {
                        "type": "table",
                        "title": "Statistics",
                        "data": {
                            "columns": [
                                {"header": "Metric", "field": "metric"},
                                {"header": "Value", "field": "value"}
                            ],
                            "rows": [{"metric": k, "value": str(v)} for k, v in stats.items()]
                        }
                    },
                    {
                        "type": "plotly",
                        "title": f"Distribution of {col}",
                        "data": [{"x": user_data[col].dropna().tolist(), "type": "histogram"}],
                        "layout": {"title": col, "height": 350}
                    }
                ]
            }],
            "next_inputs": [
                {
                    "name": "column",
                    "label": "Analyze Another Column",
                    "type": "category",
                    "constraints": {"options": list(user_data.columns)}
                }
            ]
        }

    return {
        "outputs": [{"type": "section", "title": "Error", "items": [{"type": "error", "error": "Unexpected state"}]}],
        "next_inputs": [{"name": "message", "label": "Message", "type": "string"}]
    }
```
