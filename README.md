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

API is allowed to call 10000 times per 10 minutes. If you create/delete a large size of tasks, please wait a second for every requests.

## Task

### Image

Supported following project types:

- Image - Bounding Box
- Image - Polygon
- Image - Keypoint
- Image - Line
- Image - Segmentation
- Image - All

#### Create Task

- Create a new task.

```python
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg"
)
```

- Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    annotations=[{
        "type": "bbox",
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

> Check [examples/create_image_task.py](/examples/create_image_task.py).

#### Find Task

- Find a single task.

```python
task = client.find_image_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

- Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
```

- Filter and Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_tasks(
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

    tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG", offset=offset)
    all_tasks.extend(tasks)

    if len(tasks) > 0:
        offset = len(all_tasks)  # Set the offset
    else:
        break
```

> Please wait a second before sending another requests!

#### Response

- Example of a single image task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "width": 100,   # image width
    "height": 100,  # image height
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "customerReviewer": "CUSTOMER_REVIEWER_NAME",
    "annotations": [
        {
            "attributes": [
                { "key": "kind", "name": "Kind", "type": "text", "value": "Scottish field" }
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

### Image Classification

Supported following project types:

- Image - Classification

#### Create Task

- Create a new task.

```python
task_id = client.create_image_classification_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

#### Find Task

- Find a single task.

```python
task = client.find_image_classification_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

- Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Response

- Example of a single image classification task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "width": 100,   # image width
    "height": 100,  # image height
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "customerReviewer": "CUSTOMER_REVIEWER_NAME",
    "attributes": [
        {
            "key": "kind",
            "name": "Kind",
            "type": "text",
            "value": "Scottish field"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Multi Image

Supported following project types:

- Multi Image - Bounding Box
- Multi Image - Polygon
- Multi Image - Keypoint
- Multi Image - Line
- Multi Image - Segmentation

#### Create Task

- Create a new task.

```python
task = client.create_multi_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample",
    folder_path="./sample",
    annotations=[{
        "type": "segmentation",
        "value": "annotation-value",
        "attributes": [
            {
                "key": "attribute-key",
                "value": "attribute-value"
            }
        ],
        "content": "01.jpg",
        "points": [[[
            100,
            100,
            300,
            100,
            300,
            300,
            100,
            300,
            100,
            100
        ]]] # clockwise rotation
    }]
)
```

#### Find Task

- Find a single task.

```python
task = client.find_multi_image_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

- Get tasks.

```python
tasks = client.get_multi_image_tasks(project="YOUR_PROJECT_SLUG")
```

#### Response

- Example of a single task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "contents": [
        {
            "name": "content-name",
            "url": "content-url",
            "width": 100,
            "height": 100,
        }
    ],
    "status": "registered",
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "customerReviewer": "CUSTOMER_REVIEWER_NAME",
    "annotations": [
        {
            "content": "content-name"
            "attributes": [],
            "color": "#b36d18",
            "points": [[[
                100,
                100,
                300,
                100,
                300,
                300,
                100,
                300,
                100,
                100
            ]]]
            "title": "Cat",
            "type": "bbox",
            "value": "cat"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Video

Supported following project types:

- Video - Bounding Box

#### Create Task

- Create a new task.

```python
task_id = client.create_video_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp4",
    file_path="./sample.mp4"
)
```

#### Find Task

- Find a single task.

```python
task = client.find_video_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

- Get tasks. (Up to 10 tasks)

```python
tasks = client.get_video_tasks(project="YOUR_PROJECT_SLUG")
```

#### Response

- Example of a single image classification task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "width": 100,   # image width
    "height": 100,  # image height
    "fps": 30.0,    # frame per seconds
    "frameCount": 480,  # total frame count of video
    "duration": 16.0,   # total duration of video
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "customerReviewer": "CUSTOMER_REVIEWER_NAME",
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "points": {
                "1": {  # number of frame
                    "value": [
                        100,  # top-left x
                        100,  # top-left y
                        200,  # bottom-right x
                        200   # bottom-right y
                    ],
                    "autogenerated": False  # False when annotated manually. True when auto-generated by system.
                },
                "2": {
                    "value": [
                        110,
                        110,
                        220,
                        220
                    ],
                    "autogenerated": True
                },
                "3": {
                    "value": [
                        120,
                        120,
                        240,
                        240
                    ],
                    "autogenerated": False
                }
            },
            "title": "Cat",
            "type": "bbox",
            "value": "cat"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Common

APIs for update and delete are same over all tasks.

#### Update Task

- Update a single task status and tags.

```python
task_id = client.update_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    tags=["tag1", "tag2"]
)
```

#### Delete Task

- Delete a single task.

```python
client.delete_task(task_id="YOUR_TASK_ID")
```

## Converter

### COCO

- Get tasks and convert to [COCO format](https://cocodataset.org/#format-data) (supporting bbox or polygon annotation type).

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
pprint(client.to_coco(tasks))
```

## API Docs

Check [this](https://api.fastlabel.ai/docs/) for further information.
