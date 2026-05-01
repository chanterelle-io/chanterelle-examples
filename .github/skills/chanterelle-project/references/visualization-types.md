# Visualization Types Reference

All project types (model findings, analytics, interactive outputs) share the same visualization system for rendering content. Items live inside sections.

## Sections

```json
{
  "type": "section",
  "id": "unique-id",
  "title": "Section Title",
  "description": "Optional description",
  "color": "green",
  "items_per_row": 2,
  "collapsible": true,
  "collapsed": false,
  "items": [...]
}
```

Sections can nest other sections. Use `items_per_row` for grid layout.

### Dropdown Sections

Show different content based on user selection:

```json
{
  "type": "section",
  "title": "Choose View",
  "dropdown": {
    "enabled": true,
    "default_selection": "option1",
    "options": [
      {"id": "option1", "label": "Option 1"},
      {"id": "option2", "label": "Option 2"}
    ]
  },
  "subsections": {
    "option1": {"items_per_row": 1, "items": [...]},
    "option2": {"items_per_row": 2, "items": [...]}
  }
}
```

## Common Item Properties

All items support:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | **Required.** The visualization type |
| `id` | string | Unique identifier |
| `title` | string | Display title |
| `description` | string | Help text |
| `comment` | string | Additional commentary |
| `collapsible` | boolean | Can be collapsed |
| `collapsed` | boolean | Starts collapsed |
| `full_width` | boolean | Override items_per_row, take full width |

## Item Types

### table

```json
{
  "type": "table",
  "title": "Results",
  "data": {
    "columns": [
      {"header": "Name", "field": "name"},
      {"header": "Score", "field": "score"}
    ],
    "rows": [
      {"name": "Model A", "score": 0.95},
      {"name": "Model B", "score": 0.87}
    ]
  }
}
```

### bar_chart

```json
{
  "type": "bar_chart",
  "title": "Feature Importance",
  "data": {
    "bars": [
      {"label": "Feature A", "value": 0.85},
      {"label": "Feature B", "value": 0.72}
    ],
    "axis": {
      "x": {"label": "Features"},
      "y": {"label": "Importance"}
    }
  }
}
```

### line_chart

```json
{
  "type": "line_chart",
  "title": "Training Loss",
  "data": {
    "lines": [
      {
        "id": "train",
        "points": [{"x": 1, "y": 0.9}, {"x": 2, "y": 0.5}, {"x": 3, "y": 0.3}],
        "style": {"color": "blue", "width": 2}
      }
    ],
    "axis": {
      "x": {"label": "Epoch"},
      "y": {"label": "Loss"}
    }
  }
}
```

### scatter_plot

```json
{
  "type": "scatter_plot",
  "title": "Feature Scatter",
  "data": {
    "points": [
      {"x": 1.2, "y": 3.4},
      {"x": 2.5, "y": 1.8}
    ],
    "axis": {
      "x": {"label": "Feature A"},
      "y": {"label": "Feature B"}
    }
  }
}
```

### text

Rich text with paragraphs and bullet lists:

```json
{
  "type": "text",
  "title": "Summary",
  "content": [
    {"type": "paragraph", "text": "Key findings:"},
    {
      "type": "bullet_list",
      "items": [
        {"text": "Point 1"},
        {"text": "Point 2", "style": {"bold": true}},
        {"text": "Point 3", "items": [{"text": "Sub-point"}]}
      ]
    }
  ]
}
```

Paragraph style options: `color`, `font_size`, `bold`, `italic`.

### image

```json
{
  "type": "image",
  "title": "Confusion Matrix",
  "file_path": "graphs/confusion_matrix.png",
  "caption": "Optional caption"
}
```

Paths: relative to project dir, absolute, `file://`, or `http(s)://`.

### markdown

GFM Markdown with syntax-highlighted code blocks:

```json
{
  "type": "markdown",
  "title": "Analysis Notes",
  "content": "# Heading\n\n- Bullet\n\n```python\nprint('hello')\n```"
}
```

Or from file:

```json
{
  "type": "markdown",
  "title": "Report",
  "file_path": "docs/report.md"
}
```

### plotly

Full Plotly.js charts. Three ways to provide data:

**Inline data + layout:**
```json
{
  "type": "plotly",
  "title": "Scatter",
  "data": [
    {"x": [1,2,3], "y": [2,6,3], "type": "scatter", "mode": "lines+markers"}
  ],
  "layout": {"title": "My Chart", "height": 400}
}
```

**From Plotly JSON file:**
```json
{
  "type": "plotly",
  "title": "From File",
  "file_path": "graphs/chart.json"
}
```

**Stringified JSON (e.g., from `plotly.io.to_json()`):**
```json
{
  "type": "plotly",
  "title": "Figure",
  "figure_json": "{\"data\": [...], \"layout\": {...}}"
}
```

### html

Embedded HTML in a sandboxed iframe:

```json
{
  "type": "html",
  "title": "Custom Viz",
  "content": "<h1>Hello</h1><p>Inline HTML</p>"
}
```

Or from file:

```json
{
  "type": "html",
  "title": "Interactive Viz",
  "file_path": "graphs/visualization.html"
}
```

### error

```json
{
  "type": "error",
  "title": "Processing Error",
  "error": "File not found: data.csv"
}
```

## External References ($href)

Inline content from external JSON files:

```json
{"$href": "graphs/chart_data.json"}
```

The referenced file's content replaces the `$href` object. Works in `content` arrays and `items` arrays. Resolved recursively and relative to the project directory.
