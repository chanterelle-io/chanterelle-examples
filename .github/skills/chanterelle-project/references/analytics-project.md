# Analytics Project Reference

Analytics projects are JSON-only dashboards — no Python required.

## analytics.json Schema

```json
{
  "analysis_name": "Display Name",
  "version": "1.0.0",
  "description_short": "Brief description for catalog card",
  "content": [
    {
      "type": "section",
      "id": "section-id",
      "title": "Section Title",
      "description": "Optional description",
      "items_per_row": 2,
      "items": [...]
    }
  ]
}
```

### Top-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `analysis_name` | Yes | Name shown in catalog and page header |
| `version` | No | Version string |
| `description_short` | No | Short text for catalog card |
| `content` | Yes | Array of sections |

### Content

The `content` array contains sections. Sections can be nested and support all visualization types documented in [visualization-types.md](./visualization-types.md).

Section features:
- `items_per_row` — grid layout (1, 2, 3, etc.)
- `dropdown` + `subsections` — filterable content with tabs
- `collapsible` / `collapsed` — collapsible sections
- `color` — colored section headers
- `$href` — inline external JSON files

## Example: Dashboard with Images and Plotly

```json
{
  "analysis_name": "Sales Analysis Q4",
  "version": "1.0.0",
  "description_short": "Quarterly sales performance dashboard",
  "content": [
    {
      "type": "section",
      "id": "overview",
      "title": "Overview",
      "items_per_row": 2,
      "items": [
        {
          "type": "image",
          "title": "Revenue by Region",
          "file_path": "graphs/revenue_by_region.png"
        },
        {
          "type": "plotly",
          "title": "Monthly Trend",
          "file_path": "graphs/monthly_trend.json"
        }
      ]
    },
    {
      "type": "section",
      "id": "details",
      "title": "Detailed Metrics",
      "items": [
        {"$href": "graphs/metrics_table.json"}
      ]
    }
  ]
}
```

## Example: Dropdown Sections

```json
{
  "type": "section",
  "id": "charts",
  "title": "Charts",
  "dropdown": {
    "enabled": true,
    "default_selection": "bar",
    "options": [
      {"id": "bar", "label": "Bar Charts"},
      {"id": "line", "label": "Line Charts"}
    ]
  },
  "subsections": {
    "bar": {
      "items_per_row": 2,
      "items": [
        {"type": "bar_chart", "title": "Sales", "data": {"bars": [{"label": "Q1", "value": 100}], "axis": {"x": {"label": "Quarter"}, "y": {"label": "Revenue"}}}}
      ]
    },
    "line": {
      "items_per_row": 1,
      "items": [
        {"type": "line_chart", "title": "Trend", "data": {"lines": [{"id": "main", "points": [{"x": 1, "y": 10}, {"x": 2, "y": 20}]}], "axis": {"x": {"label": "Month"}, "y": {"label": "Value"}}}}
      ]
    }
  }
}
```
