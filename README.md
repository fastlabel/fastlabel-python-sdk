# FastLabel Python SDK

## Installation

```bash
$ pip install --upgrade fastlabel
```

> Python version 3.7 or greater is required

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

## Limitation

API is allowed to call 5000 times per hour. If you create/delete a large size of tasks, please wait a second for every requests.

## Task

### Create Task

- Create a new task.

```python
task = client.create_image_task(
    project_id="YOUR_PROJECT_ID",
    key="sample.jpg",
    url="https://sample.com/sample.jpg"
)
```

- Create a new task with pre-defined labels. (Class should be configured on your project in advance)

```python
task = client.create_image_task(
    project_id="YOUR_PROJECT_ID",
    key="sample.jpg",
    url="https://sample.com/sample.jpg",
    labels=[
        {
            "type": "bbox",
            "value": "bbox",
            "points": [
                { "x": 100, "y": 100},  # top-left
                { "x": 200, "y": 200}   # bottom-right
            ]
        }
    ]
)
```

> Check [examples/create_image_task.py](/examples/create_image_task.py) for other label types, such as line, keyPoint and polygon.

### Find Task

- Find a single task.

```python
task = client.find_task(task_id="YOUR_TASK_ID")
```

### Get Tasks

- Get tasks. (Up to 100 tasks)

```python
tasks = client.get_tasks(project_id="YOUR_PROJECT_ID")
```

- Filter and Get tasks. (Up to 100 tasks)

```python
tasks = client.get_tasks(
    project_id="YOUR_PROJECT_ID",
    status="submitted", # status can be 'registered', 'registered', 'submitted' or 'skipped'
    review_status="accepted" # review_status can be 'notReviewed', 'inProgress', 'accepted' or 'declined'
)
```

- Get a large size of tasks. (Over 100 tasks)

```python
import time

# Iterate pages until new tasks are empty.
all_tasks = []
start_after = None
while True:
    time.sleep(1)

    tasks = client.get_tasks(project_id="YOUR_PROJECT_ID", start_after=start_after)
    all_tasks.extend(tasks)

    if len(tasks) > 0:
        start_after = tasks[-1]["id"] # Set the last task id to start_after
    else:
        break
```

> Please wait a second before sending another requests!

### Delete Task

```python
client.delete_task(task_id="YOUR_TASK_ID")
```

### Task Response

- Example of a single task object

```python
{
    "id": "YOUR_TASK_ID",
    "key": "sample.png",
    "assigneeId": null,
    "assigneeName": null,
    "status": "registered",
    "reviewAssigneeId": null,
    "reviewAssigneeName": null,
    "reviewStatus": "notReviewed",
    "projectId": "YOUR_PROJECT_ID",
    "datasetId": "YOUR_DATASET_ID",
    "labels": [
        {
            "id": "YOUR_LABEL_ID",
            "type": "bbox",
            "value": "window",
            "title": "窓",
            "color": "#d9713e",
            "metadata": [],
            "points": [
                { "x": 100, "y": 100},  # top-left
                { "x": 200, "y": 200}   # bottom-right
            ]
        }
    ],
    "duration": 0,
    "image": {
        "width": 1500,
        "height": 1200
    },
    "createdAt": "2020-12-25T15:02:00.513",
    "updatedAt": "2020-12-25T15:02:00.513"
}
```

## Model Analysis

### Upload Predictions

```python
# Create your model predictions
predictions = [
    {
        "fileKey": "sample.jpg",  # file name exists in your project
        "labels": [
            {
                "value": "bbox_a",  # class value exists in your project
                "points": [
                    { "x": 10, "y": 10 },   # top-left
                    { "x": 20, "y": 20 },   # botom-right
                ]
            },
            {
                "value": "bbox_b",
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
    project_id="YOUR_PROJECT_ID",    # your fastlabel project id
    analysis_type="bbox",    # annotation type to be analyze (Only "bbox" or "line" are supported)
    threshold=80,   # IoU percentage/pixel distance to check labels are correct. (Ex: 0 - 100)
    predictions=predictions
)
```

## API Docs

Check [this](https://api-fastlabel-production.web.app/api/doc/) for further information.
