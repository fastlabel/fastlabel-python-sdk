# FastLabel Python SDK

_If you are using FastLabel prototype, please install version 0.2.2._

## Installation

```bash
$ pip install --upgrade fastlabel
```

> Python version 3.7 or greater is required

## Usage

Configure API Key in environment variable.

```bash
export FASTLABEL_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
```

Initialize fastlabel client.

```python
import fastlabel
client = fastlabel.Client()
```

## Limitation

API is allowed to call 1000 times per 10 minutes. If you create/delete a large size of tasks, please wait a second for every requests.

## Task

### Create Task

- Create a new task.

```python
task_id = client.create_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg"
)
```

- Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    annotations=[{
        "value": "annotation-value",
        "attributes": [
            {
                "key": "attribute-key",
                "value": "attribute-value"
            }
        ],
        "points": [
            100,  # top-left x
            100,  # top-left y
            200,  # bottom-right x
            200   # bottom-right y
        ]
    }]
)
```

> Check [examples/create_task.py](/examples/create_task.py).

### Update Task

- Update a single task status, tags, and annotations.

```python
task_id = client.update_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    tags=["tag1", "tag2"],
    annotations=[{
        "value": "annotation-value",
        "attributes": [
            {
                "key": "attribute-key",
                "value": "attribute-value"
            }
        ],
        "points": [
            100,  # top-left x
            100,  # top-left y
            200,  # bottom-right x
            200   # bottom-right y
        ]
    }]
)
```

### Find Task

- Find a single task.

```python
task = client.find_task(task_id="YOUR_TASK_ID")
```

### Get Tasks

- Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_tasks(project="YOUR_PROJECT_SLUG")
```

- Filter and Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_tasks(
    project="YOUR_PROJECT_SLUG",
    status="approved", # status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'
    tags=["tag1", "tag2"] # up to 10 tags
)
```

- Get a large size of tasks. (Over 1000 tasks)

```python
import time

# Iterate pages until new tasks are empty.
all_tasks = []
offset = None
while True:
    time.sleep(1)

    tasks = client.get_tasks(
        project="YOUR_PROJECT_SLUG", offset=offset)
    all_tasks.extend(tasks)

    if len(tasks) > 0:
        offset = len(all_tasks)  # Set the offset
    else:
        break
```

> Please wait a second before sending another requests!

### Delete Task

- Delete a single task.

```python
client.delete_task(task_id="YOUR_TASK_ID")
```

### Task Response

- Example of a single task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "tags": [],
    "annotations": [
        {
            "attributes": [
                { "key": "kind", "name": "猫の種類", "type": "text", "value": "三毛猫" }
            ],
            "color": "#b36d18",
            "points": [
                100,  # top-left x
                100,  # top-left y
                200,  # bottom-right x
                200   # bottom-right y
            ],
            "title": "Cat",
            "type": "bbox",
            "value": "cat"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

## API Docs

Check [this](https://api.fastlabel.ai/docs/) for further information.
