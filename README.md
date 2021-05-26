# FastLabel Python SDK

_If you are using FastLabel prototype, please install version 0.2.2._

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Limitation](#limitation)
- [Task](#task)
  - [Image](#image)
  - [Image Classification](#image-classification)
  - [Multi Image](#multi-image)
  - [Video](#video)
  - [Common](#common)
- [Annotation](#annotation)
- [Converter](#converter)
  - [COCO](#coco)
  - [YOLO](#yolo)
  - [Pascal VOC](#pascal-voc)

## Installation

```bash
pip install --upgrade fastlabel
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

### Limitation

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

- Find a single task by name.

```python
tasks = client.find_image_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
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

#### Get Tasks Id and Name map

```python
map = client.get_task_id_name_map(project="YOUR_PROJECT_SLUG")
```

## Annotation

### Create Annotaion

- Create a new annotation.

```python
annotation_id = client.create_annotation(
    project="YOUR_PROJECT_SLUG", type="bbox", value="cat", title="Cat")
```

- Create a new annotation with color and attributes.

```python
attributes = [
    {
        "type": "text",
        "name": "Kind",
        "key": "kind"
    },
    {
        "type": "select",
        "name": "Size",
        "key": "size",
        "options": [ # select, radio and checkbox type requires options
            {
                "title": "Large",
                "value": "large"
            },
            {
                "title": "Small",
                "value": "small"
            },
        ]
    },
]
annotation_id = client.create_annotation(
    project="YOUR_PROJECT_SLUG", type="bbox", value="cat", title="Cat", color="#FF0000", attributes=attributes)
```

- Create a new classification annotation.

```python
annotation_id = client.create_classification_annotation(
    project="YOUR_PROJECT_SLUG", attributes=attributes)
```

### Find Annotation

- Find an annotation.

```python
annotaion = client.find_annotation(annotation_id="YOUR_ANNOTATIPN_ID")
```

- Find an annotation by value.

```python
annotaion = client.find_annotation_by_value(project="YOUR_PROJECT_SLUG", value="cat")
```

- Find an annotation by value in classification project.

```python
annotaion = client.find_annotation_by_value(
    project="YOUR_PROJECT_SLUG", value="classification") # "classification" is fixed value
```

### Get Annotations

- Get annotations. (Up to 1000 annotations)

```python
annotatios = client.get_annotations(project="YOUR_PROJECT_SLUG")
```

### Response

- Example of an annotation object

```python
{
    "id": "YOUR_ANNOTATION_ID",
    "type": "bbox",
    "value": "cat",
    "title": "Cat",
    "color": "#FF0000",
    "attributes": [
        {
            "id": "YOUR_ATTRIBUTE_ID",
            "key": "kind",
            "name": "Kind",
            "options": [],
            "type": "text",
            "value": ""
        },
        {
            "id": "YOUR_ATTRIBUTE_ID",
            "key": "size",
            "name": "Size",
            "options": [
                {"title": "Large", "value": "large"},
                {"title": "Small", "value": "small"}
            ],
            "type": "select",
            "value": ""
        }
    ],
    "createdAt": "2021-05-25T05:36:50.459Z",
    "updatedAt": "2021-05-25T05:36:50.459Z"
}
```

### Update Annotation

- Update an annotation.

```python
annotation_id = client.update_annotation(
    annotation_id="YOUR_ANNOTATION_ID", value="cat2", title="Cat2", color="#FF0000")
```

- Update an annotation with attributes.

```python
attributes = [
    {
        "id": "YOUR_ATTRIBUTE_ID",  # check by sdk get methods
        "type": "text",
        "name": "Kind2",
        "key": "kind2"
    },
    {
        "id": "YOUR_ATTRIBUTE_ID",
        "type": "select",
        "name": "Size2",
        "key": "size2",
        "options": [
            {
                "title": "Large2",
                "value": "large2"
            },
            {
                "title": "Small2",
                "value": "small2"
            },
        ]
    },
]
annotation_id = client.update_annotation(
    annotation_id="YOUR_ANNOTATION_ID", value="cat2", title="Cat2", color="#FF0000", attributes=attributes)
```

- Update a classification annotation.

```python
annotation_id = client.update_classification_annotation(
    project="YOUR_PROJECT_SLUG", attributes=attributes)
```

### Delete Annotation

- Delete an annotation.

```python
client.delete_annotation(annotation_id="YOUR_ANNOTATIPN_ID")
```

## Converter

Supporting bbox or polygon annotation type.

### COCO

- Get tasks and export as a [COCO format](https://cocodataset.org/#format-data) file.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_coco(tasks)
```

- Export with specifying output directory.

```python
client.export_coco(tasks=tasks, output_dir="YOUR_DIRECTROY")
```

### YOLO

- Get tasks and export as YOLO format files.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_yolo(tasks)
```

### Pascal VOC

- Get tasks and export as Pascal VOC format files.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_pascalvoc(tasks)
```

## API Docs

Check [this](https://api.fastlabel.ai/docs/) for further information.
