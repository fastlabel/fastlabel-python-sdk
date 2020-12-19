# FastLabel Python SDK

## Installation

```bash
$ pip install --upgrade fastlabel
```

> Python version 3.8 or greater is required

## Usage

Configure API Key in environment variable.

```bash
export FASTLABEL_API_KEY="YOUR_API_KEY"
```

Initialize fastlabel client.

```python
import fastlabel
client = fastlabel.Client()
```

## Model Analysis

### Upload Predictions

```python
import fastlabel

# Initialize client
client = fastlabel.Client()

# Create predictions
predictions = [
    {
        "fileKey": "sample1.jpg",  # file name exists in your project
        "labels": [
            {
                "value": "line_a",  # class value exists in your project
                "points": [
                    { "x": 10, "y": 10 },
                    { "x": 20, "y": 20 },
                ]
            },
            {
                "value": "line_b",
                "points": [
                    { "x": 30, "y": 30 },
                    { "x": 40, "y": 40 },
                ]
            }
        ]
    }
]

# Upload predictions
client.upload_predictions(
    project_id="project_id",    # your fastlabel project id
    analysis_type="line",    # annotation type to be analyze ("bbox" or "line" are supported)
    threshold=25,   # IoU percentage/pixel to analyze labels. (Ex: 0 - 100)
    predictions=predictions
)
```

## API Docs

Check [this](https://api-fastlabel-production.web.app/api/doc/) for further information.
