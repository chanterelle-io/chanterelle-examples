# Model Project Reference

## model_meta.json Schema

```json
{
  "model_id": "unique-id",
  "model_name": "Display Name",
  "model_version": "1.0.0",
  "description_short": "One-line summary for catalog card",
  "description": "Detailed description for model page",
  "links": [
    {"name": "Link Name", "url": "https://..."},
    {"name": "Local Doc", "file_name": "docs/guide.pdf"}
  ],
  "tags": {"domain": "nlp", "task": "classification"},
  "inputs": [],
  "outputs": [],
  "input_presets": [],
  "input_groupings": [],
  "python_environment": {"type": "system"},
  "allow_feedback": false
}
```

### Inputs

Each input generates a form field. Required fields: `name`, `label`, `type`.

```json
{
  "name": "param_name",
  "label": "Display Label",
  "type": "float",
  "required": true,
  "description": "Help text",
  "default": 1.0,
  "unit": "meters",
  "constraints": {},
  "depends_on": {}
}
```

#### Input Types and Constraints

| Type | Constraint Fields |
|------|------------------|
| `float` | `min`, `max`, `step` |
| `int` | `min`, `max`, `step` |
| `string` | `regex`, `placeholder` |
| `category` | `options` (string[] or `{value, label, description}[]`) |
| `boolean` | *(none)* |
| `textarea` | `rows`, `placeholder` |
| `file` | `extensions` (string[]), `multiple` (bool) |
| `button` | `placeholder` (button text) |
| `yes_no` | `yes_label`, `no_label`, `yes_value`, `no_value` |

#### Conditional Inputs (depends_on)

Constraints change based on another input's value:

```json
{
  "name": "strength",
  "type": "float",
  "depends_on": {
    "input_name": "algorithm",
    "mapping": {
      "svm": {"constraints": {"min": 0.001, "max": 10.0}},
      "logistic": {"constraints": {"min": 0.01, "max": 100.0}}
    }
  }
}
```

### Outputs

```json
{
  "name": "result",
  "label": "Display Label",
  "type": "float",
  "description": "What this output represents",
  "unit": "USD",
  "min": 0,
  "max": 100,
  "options": [{"value": "a", "label": "Option A"}]
}
```

Output types: `float`, `int`, `string`, `boolean`.

### Input Presets

Pre-fill multiple inputs at once:

```json
{
  "input_preset": "preset_group_id",
  "label": "Preset Group Name",
  "description": "Help text",
  "affects": ["input1", "input2"],
  "presets": [
    {"name": "Fast", "values": {"input1": 10, "input2": 0.1}},
    {"name": "Accurate", "values": {"input1": 100, "input2": 0.01}}
  ]
}
```

### Input Groupings

Group inputs into fieldsets:

```json
{
  "grouping": "group_id",
  "description": "Group description",
  "inputs": ["input1", "input2"]
}
```

## handler_io.py

Model handlers implement 4 required + 1 optional function:

```python
def model_fn(model_dir):
    """Required. Load and return the model object."""
    import joblib, os
    return joblib.load(os.path.join(model_dir, 'model.joblib'))

def init_resources_fn(model_dir):
    """Optional. Load extra resources (lookups, configs)."""
    import json, os
    with open(os.path.join(model_dir, 'config.json')) as f:
        return {"config": json.load(f)}

def input_fn(request_data, resources=None):
    """Required. Transform raw input dict into model-ready data."""
    return [[
        request_data['feature1'],
        request_data['feature2'],
    ]]

def predict_fn(input_data, model, resources=None):
    """Required. Run prediction, return raw results."""
    return model.predict(input_data)

def output_fn(predictions, original_data, resources=None):
    """Required. Format predictions as a list of sections.
    Returns the same structure as model_findings.json content array.
    See visualization-types.md for all supported item types.
    """
    return [
        {
            "type": "section",
            "id": "results",
            "title": "Prediction Results",
            "items": [
                {
                    "type": "table",
                    "id": "pred_table",
                    "title": "Results",
                    "data": {
                        "columns": [
                            {"header": "Metric", "field": "metric"},
                            {"header": "Value", "field": "value"}
                        ],
                        "rows": [
                            {"metric": "Prediction", "value": str(predictions[0])}
                        ]
                    }
                }
            ]
        }
    ]
```

## model_findings.json (Optional)

Static insights displayed on the Insights tab:

```json
{
  "model_id": "my-model",
  "version": "1.0.0",
  "content": [
    {
      "type": "section",
      "id": "section1",
      "title": "Section Title",
      "items": [...]
    }
  ]
}
```

See [visualization-types.md](./visualization-types.md) for all supported item types.

Use `{"$href": "path/to/file.json"}` to inline external JSON content.
