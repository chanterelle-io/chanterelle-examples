---
name: chanterelle-project
description: 'Create, scaffold, or modify Chanterelle desktop app projects. Use when: building ML model interfaces, creating analytics dashboards, building interactive conversational agents, writing handler_io.py, authoring model_meta.json, analytics.json, interactive.json, or model_findings.json. Covers all 3 project types: model, analytics, interactive.'
argument-hint: 'Describe the project type (model/analytics/interactive) and what it should do'
---

# Chanterelle Project Builder

Build projects for the Chanterelle desktop app — a local ML project catalog with model interfaces, analytics dashboards, and interactive agents.

## Project Types

| Type | Identified By | Requires Python | Description |
|------|--------------|-----------------|-------------|
| **Model** | `model_meta.json` | Yes | Test & showcase ML models via generated forms |
| **Analytics** | `analytics.json` | No | Static insight dashboards with rich visualizations |
| **Interactive** | `interactive.json` | Yes | Multi-turn conversational agents with dynamic forms |

## Procedure

### 1. Determine project type
Ask the user what they want to build if unclear. Use the table above to match.

### 2. Scaffold the project
Create the required files based on project type:

**Model project:**
```
project_name/
    model_meta.json          # UI config: inputs, outputs, presets, groupings
    handler_io.py            # Python: model_fn, input_fn, predict_fn, output_fn
    model_findings.json      # Optional: static insight report
```

**Analytics project:**
```
project_name/
    analytics.json           # Sections with visualizations
    graphs/                  # Optional: images, Plotly JSON, HTML files
```

**Interactive project:**
```
project_name/
    interactive.json         # Agent metadata and Python env config
    handler_io.py            # Python: initialize() and on_input(data)
```

### 3. Write the configuration files
Load the appropriate reference for the project type:

- Model: [model_meta.json schema](./references/model-project.md) and [handler_io.py pattern](./references/model-project.md#handler_iopy)
- Analytics: [analytics.json schema](./references/analytics-project.md)
- Interactive: [interactive.json schema](./references/interactive-project.md) and [handler_io.py pattern](./references/interactive-project.md#handler_iopy)

### 4. Add visualizations
All project types share the same visualization system for output/insight content. Load the [visualization types reference](./references/visualization-types.md) when building sections and items.

### 5. Configure Python environment
For model and interactive projects, set `python_environment`:

```json
{"type": "system"}
{"type": "venv", "path": "../my-env"}
{"type": "conda", "name": "my-env"}
{"type": "virtualenv", "path": "../my-env"}
```

## Key Constraints

- All file paths in JSON are **relative to the project directory**
- Inputs support 9 types: `float`, `int`, `string`, `category`, `boolean`, `textarea`, `file`, `button`, `yes_no`
- The `$href` keyword in JSON inlines content from external files (resolved recursively)
- Model `output_fn` and interactive `on_input` return the same section/item format as `model_findings.json`
- Interactive handlers maintain state via module-level Python variables (process persists across turns)
