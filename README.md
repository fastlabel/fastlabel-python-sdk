# FastLabel Python SDK

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Limitation](#limitation)
- [Task](#task)
  - [Image](#image)
  - [Image Classification](#image-classification)
  - [Multi Image Classification](#multi-image-classification)
  - [Sequential Image](#sequential-image)
  - [Video](#video)
  - [Video Classification](#video-classification)
  - [Text](#text)
  - [Text Classification](#text-classification)
  - [Audio](#audio)
  - [Audio Classification](#audio-classification)
  - [PCD](#pcd)
  - [Sequential PCD](#sequential-pcd)
  - [DICOM](#dicom)
  - [Common](#common)
- [Annotation](#annotation)
- [Project](#project)
- [Dataset](#dataset)
- [Converter](#converter)
  - [FastLabel To COCO](#fastlabel-to-coco)
  - [FastLabel To YOLO](#fastlabel-to-yolo)
  - [FastLabel To Pascal VOC](#fastlabel-to-pascal-voc)
  - [FastLabel To labelme](#fastlabel-to-labelme)
  - [FastLabel To Segmentation](#fastlabel-to-segmentation)
  - [COCO To FastLabel](#coco-to-fastlabel)
  - [YOLO To FastLabel](#yolo-to-fastlabel)
  - [Pascal VOC To FastLabel](#pascal-voc-to-fastlabel)
  - [labelme To FastLabel](#labelme-to-fastlabel)
- [Model](#model)
- [API Docs](#api-docs)

## Installation

```bash
pip install --upgrade fastlabel
```

> Python version 3.8 or greater is required

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
- Image - Pose Estimation
- Image - All

#### Create Task

Create a new task.

```python
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg"
)
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
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

##### Limitation

- You can upload up to a size of 20 MB.

#### Create Integrated Image Task

Create a new task by integrated image.
(Project storage setting should be configured in advance.)

```python
task_id = client.create_integrated_image_task(
    project="YOUR_PROJECT_SLUG",
    file_path="<integrated-storage-dir>/sample.jpg",
    storage_type="gcp",
)
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    file_path="<integrated-storage-dir>/sample.jpg",
    storage_type="gcp",
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

##### Limitation

- You can upload up to a size of 20 MB.

#### Find Task

Find a single task.

```python
task = client.find_image_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_image_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
```

- Filter and Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_tasks(
    project="YOUR_PROJECT_SLUG",
    status="approved", # status can be 'pending', 'registered', 'completed', 'skipped', 'reviewed' 'sent_back', 'approved', 'declined'
    tags=["tag1", "tag2"] # up to 10 tags
)
```

Get a large size of tasks. (Over 1000 tasks)

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

#### Update Tasks

Update a single task.

```python
task_id = client.update_image_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    annotations=[
        {
            "type": "bbox",
            "value": "cat"
            "attributes": [
                { "key": "kind", "value": "Scottish field" }
            ],
            "points": [
                100,  # top-left x
                100,  # top-left y
                200,  # bottom-right x
                200   # bottom-right y
            ]
        }
    ],
    # pass annotation indexes to update
    relations=[
      {
        "startIndex": 1,
        "endIndex": 0,
      },
      {
        "startIndex": 2,
        "endIndex": 0
      }
    ]
)
```

#### Response

Example of a single image task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "width": 100,   # image width
    "height": 100,  # image height
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
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
            "rotation": 0,
            "title": "Cat",
            "type": "bbox",
            "value": "cat"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

Example when the project type is Image - Pose Estimation

```python
{
    "id": "YOUR_TASK_ID",
    "name": "person.jpg",
    "width": 255,   # image width
    "height": 255,  # image height
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
    "annotations":[
       {
          "type":"pose_estimation",
          "title":"jesture",
          "value":"jesture",
          "color":"#10c414",
          "attributes": [],
          "keypoints":[
             {
                "name":"頭",
                "key":"head",
                "value":[
                   102.59, # x
                   23.04,  # y
                   1       # 0:invisible, 1:visible
                ],
                "edges":[
                   "right_shoulder",
                   "left_shoulder"
                ]
             },
             {
                "name":"右肩",
                "key":"right_shoulder",
                "value":[
                   186.69,
                   114.11,
                   1
                ],
                "edges":[
                   "head"
                ]
             },
             {
                "name":"左肩",
                "key":"left_shoulder",
                "value":[
                   37.23,
                   109.29,
                   1
                ],
                "edges":[
                   "head"
                ]
             }
          ]
       }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

#### Export Image With Annotations

Get tasks and export images with annotations.
Only support the following image extension.

- jpeg
- jpg
- png
- tif
- tiff
- bmp

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_image_with_annotations(
    tasks=tasks, image_dir="IMAGE_DIR", output_dir="OUTPUT_DIR"
)
```

#### Integrate Task

This function is alpha version. It is subject to major changes in the future.

Integration is possible only when tasks are registered from the objects divided by the dataset.
Only bbox and polygon annotation types are supported.

In the case of a task divided under the following conditions.

- Dataset slug: `image`
- Object name: `cat.jpg`
- Split count: `3×3`

Objects are registered in the data set in the following form.

- image/cat/1.jpg
- image/cat/2.jpg
- image/cat/3.jpg
- (omit)
- image/cat/9.jpg

The annotations at the edges of the image are combined. However, annotations with a maximum length of 300px may not work.

In this case, SPLIT_IMAGE_TASK_NAME_PREFIX specifies `image/cat`.

```python
task = client.find_integrated_image_task_by_prefix(
    project="YOUR_PROJECT_SLUG",
    prefix="SPLIT_IMAGE_TASK_NAME_PREFIX",
)
```

##### Response

Example of a integrated image task object

```python
{
    'name': 'image/cat.jpg',
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "confidenceScore"; -1,
            "keypoints": [],
            "points": [200,200,300,400],
            "rotation": 0,
            "title": "Bird",
            "type": "polygon",
            "value": "bird"
        }
    ],
}
```

### Image Classification

Supported following project types:

- Image - Classification

#### Create Task

Create a new task.

```python
task_id = client.create_image_classification_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

##### Limitation

- You can upload up to a size of 20 MB.

#### Create Integrated Image Classification Task

Create a new classification task by integrated image.
(Project storage setting should be configured in advance.)

```python
task_id = client.create_integrated_image_classification_task(
    project="YOUR_PROJECT_SLUG",
    file_path="<integrated-storage-dir>/sample.jpg",
    storage_type="gcp",
)
```

#### Find Task

Find a single task.

```python
task = client.find_image_classification_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_image_classification_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a single task.

```python
task_id = client.update_image_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

#### Response

Example of a single image classification task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.jpg",
    "width": 100,   # image width
    "height": 100,  # image height
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "externalStatus": "registered",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
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

### Multi Image Classification

Supported following project types:

- Multi Image - Classification

#### Create Task

Create a new task.

```python
task = client.create_multi_image_classification_task(
    project="YOUR_PROJECT_SLUG",
    name="sample",
    folder_path="./sample",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    attributes=[
        {
            "type": "text",
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ]
)
```

##### Limitation


- You can upload up to a size of 20 MB.
- You can upload up to a total size of 2 GB.
- You can upload up to 6 files in total.

#### Find Task

Find a single task.

```python
task = client.find_multi_image_classification_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_multi_image_classification_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks.

```python
tasks = client.get_multi_image_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Task

Update a single task.

```python
task_id = client.update_multi_image_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    attributes=[
        {
            "type": "text",
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ]
)
```

#### Response

Example of a single task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "sample",
    "contents": [
        {
            "name": "content-name-1",
            "url": "content-url-1",
            "width": 100,
            "height": 100,
        },
        {
            "name": "content-name-2",
            "url": "content-url-2",
            "width": 100,
            "height": 100,
        }
    ],
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
    "attributes": [
        {
            "type": "text",
            "key": "attribute-key-1",
            "value": "attribute-value-1"
        },
        {
            "type": "text",
            "key": "attribute-key-2",
            "value": "attribute-value-2"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Sequential Image

Supported following project types:

- Sequential Image - Bounding Box
- Sequential Image - Polygon
- Sequential Image - Keypoint
- Sequential Image - Line
- Sequential Image - Segmentation

#### Create Task

Create a new task.

```python
task = client.create_sequential_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample",
    folder_path="./sample",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
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

##### Limitation

- You can upload up to a size of 20 MB.
- You can upload up to a total size of 512 MB.
- You can upload up to 250 files in total.

#### Find Task

Find a single task.

```python
task = client.find_sequential_image_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_sequential_image_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks.

```python
tasks = client.get_sequential_image_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Task

Update a single task.

```python
task_id = client.update_sequential_image_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[
        {
            "type": "bbox",
            "value": "cat",
            "content": "cat1.jpg",
            "attributes": [
                { "key": "key", "value": "value1" }
            ],
            "points": [990, 560, 980, 550]
        }
    ]
)
```

#### Response

Example of a single task object

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
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
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
- Video - Keypoint
- Video - Line

#### Create Task

Create a new task.

```python
task_id = client.create_video_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp4",
    file_path="./sample.mp4"
)
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_video_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp4",
    file_path="./sample.mp4",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[{
        "type": "bbox",
        "value": "person",
        "points": {
            "1": {  # number of frame
                "value": [
                    100,  # top-left x
                    100,  # top-left y
                    200,  # bottom-right x
                    200   # bottom-right y
                ],
                # Make sure to set `autogenerated` False for the first and last frame. "1" and "3" frames in this case.
                # Otherwise, annotation is auto-completed for rest of frames when you edit.
                "autogenerated": False
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
        }
    }]
)
```

##### Limitation

- You can upload up to a size of 250 MB.
- You can upload only videos with H.264 encoding.
- You can upload only MP4 file format.

#### Find Task

Find a single task.

```python
task = client.find_video_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_video_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 10 tasks)

```python
tasks = client.get_video_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Task

Update a single task.

```python
task_id = client.update_video_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    annotations=[{
        "type": "bbox",
        "value": "bird",
        "points": {
            "1": {
                "value": [
                    100,
                    100,
                    200,
                    200
                ],
                "autogenerated": False
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
        }
    }]
)
```

#### Integrate Video

This function is alpha version. It is subject to major changes in the future.

Integration is possible only when tasks are registered from the objects divided by the dataset.

In the case of a task divided under the following conditions.

- Dataset slug: `video`
- Object name: `cat.mp4`
- Split count: `3`

Objects are registered in the data set in the following form.

- video/cat/1.mp4
- video/cat/2.mp4
- video/cat/3.mp4

In this case, SPLIT_VIDEO_TASK_NAME_PREFIX specifies `video/cat`.

```python
task = client.find_integrated_video_task_by_prefix(
    project="YOUR_PROJECT_SLUG",
    prefix="SPLIT_VIDEO_TASK_NAME_PREFIX",
)
```

#### Response

Example of a single vide task object

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
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
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

### Video Classification

Supported following project types:

- Video - Classification (Single)

#### Create Task

Create a new task.

```python
task_id = client.create_video_classification_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp4",
    file_path="./sample.mp4",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

##### Limitation

- You can upload up to a size of 250 MB.

#### Find Task

Find a single task.

```python
task = client.find_video_classification_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_video_classification_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_video_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a single task.

```python
task_id = client.update_video_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

### Text

Supported following project types:

- Text - NER

#### Create Task

Create a new task.

```python
task_id = client.create_text_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.txt",
    file_path="./sample.txt"
)
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_text_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.txt",
    file_path="./sample.txt",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[{
        "type": "ner",
        "value": "person",
        "start": 0,
        "end": 10,
        "text": "1234567890"
    }]
)
```

##### Limitation

- You can upload up to a size of 2 MB.

#### Find Task

Find a single task.

```python
task = client.find_text_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_text_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 10 tasks)

```python
tasks = client.get_text_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Task

Update a single task.

```python
task_id = client.update_text_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    annotations=[{
        "type": "bbox",
        "value": "bird",
        "start": 0,
        "end": 10,
        "text": "0123456789"
    }]
)
```

#### Response

Example of a single text task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.txt",
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "text": "0123456789",
            "start": 0,
            "end": 10,
            "title": "Cat",
            "type": "ner",
            "value": "cat"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Text Classification

Supported following project types:

- Text - Classification (Single)

#### Create Task

Create a new task.

```python
task_id = client.create_text_classification_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.txt",
    file_path="./sample.txt",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

##### Limitation

- You can upload up to a size of 2 MB.

#### Find Task

Find a single task.

```python
task = client.find_text_classification_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_text_classification_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_text_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a single task.

```python
task_id = client.update_text_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

### Audio

Supported following project types:

- Audio - Segmentation

#### Create Task

Create a new task.

```python
task_id = client.create_audio_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp3",
    file_path="./sample.mp3"
)
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_audio_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp3",
    file_path="./sample.mp3",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[{
        "type": "segmentation",
        "value": "person",
        "start": 0.4,
        "end": 0.5
    }]
)
```

##### Limitation

- You can upload up to a size of 120 MB.

#### Find Task

Find a single task.

```python
task = client.find_audio_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_audio_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 10 tasks)

```python
tasks = client.get_audio_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Task

Update a single task.

```python
task_id = client.update_audio_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    annotations=[{
        "type": "segmentation",
        "value": "bird",
        "start": 0.4,
        "end": 0.5
    }]
)
```

#### Response

Example of a single audio task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "cat.mp3",
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "start": 0.4,
            "end": 0.5,
            "title": "Bird",
            "type": "segmentation",
            "value": "bird"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

#### Integrate Task

This function is alpha version. It is subject to major changes in the future.

Integration is possible only when tasks are registered from the objects divided by the dataset.

In the case of a task divided under the following conditions.

- Dataset slug: `audio`
- Object name: `voice.mp3`
- Split count: `3`

Objects are registered in the data set in the following form.

- audio/voice/1.mp3
- audio/voice/2.mp3
- audio/voice/3.mp3

Annotations are combined when the end point specified in the annotation is the end time of the task and the start point of the next task is 0 seconds.

In this case, SPLIT_AUDIO_TASK_NAME_PREFIX specifies `audio/voice`.

```python
task = client.find_integrated_audio_task_by_prefix(
    project="YOUR_PROJECT_SLUG",
    prefix="SPLIT_AUDIO_TASK_NAME_PREFIX",
)
```

##### Response

Example of a integrated audio task object

```python
{
    'name': 'audio/voice.mp3',
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "start": 0.4,
            "end": 0.5,
            "title": "Bird",
            "type": "segmentation",
            "value": "bird"
        }
    ],
}
```

### Audio Classification

Supported following project types:

- Audio - Classification (Single)

#### Create Task

Create a new task.

```python
task_id = client.create_audio_classification_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.mp3",
    file_path="./sample.mp3",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

##### Limitation

- You can upload up to a size of 120 MB.

#### Find Task

Find a single task.

```python
task = client.find_audio_classification_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_audio_classification_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_audio_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a single task.

```python
task_id = client.update_audio_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

### PCD

Supported following project types:

- PCD - Cuboid
- PCD - Segmentation

#### Create Task

Create a new task.

```python
task_id = client.create_pcd_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.pcd",
    file_path="./sample.pcd"
)
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

Annotation Type: cuboid

```python
task_id = client.create_pcd_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.pcd",
    file_path="./sample.pcd",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[
        {
            "type": "cuboid",
            "value": "car",
            "points": [ # For cuboid, it is a 9-digit number.
                1, # Coordinate X
                2, # Coordinate Y
                3, # Coordinate Z
                1, # Rotation x
                1, # Rotation Y
                1, # Rotation Z
                2, # Length X
                2, # Length Y
                2  # Length Z
            ],
        }
    ],
)
```

Annotation Type: segmentation

```python
task_id = client.create_pcd_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.pcd",
    file_path="./sample.pcd",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[
        {
            "type": "segmentation",
            "value": "car",
            "points": [1, 2, 3, 4, 5], # For segmentation, it is an arbitrary numeric array.
        }
    ],
)
```

##### Limitation

- You can upload up to a size of 30 MB.

#### Find Task

Find a single task.

```python
task = client.find_pcd_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_pcd_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_pcd_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Task

Update a single task.

```python
task_id = client.update_pcd_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    annotations=[
        {
            "type": "cuboid",
            "value": "car",
            "points": [ # For cuboid, it is a 9-digit number.
                1, # Coordinate X
                2, # Coordinate Y
                3, # Coordinate Z
                1, # Rotation x
                1, # Rotation Y
                1, # Rotation Z
                2, # Length X
                2, # Length Y
                2  # Length Z
            ],
        }
    ],
)
```

#### Response

Example of a single PCD task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "sample.pcd",
    "url": "YOUR_TASK_URL",
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "tags": ["tag1", "tag2"],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "approver": "APPROVER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
    "externalApprover": "EXTERNAL_APPROVER_NAME",
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "title": "Car",
            "type": "segmentation",
            "value": "car",
            "points": [1, 2, 3, 1, 1, 1, 2, 2, 2],
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Sequential PCD

Supported following project types:

- Sequential PCD - Cuboid

#### Create Tasks

Create a new task.

```python
task_id = client.create_sequential_pcd_task(
    project="YOUR_PROJECT_SLUG",
    name="drive_record",
    folder_path="./drive_record/", # Path where sequence PCD files are directory
)
```

The order of frames is determined by the ascending order of PCD file names located in the specified directory.
File names are optional, but we recommend naming them in a way that makes the order easy to understand.

```
./drive_record/
├── 0001.pcd => frame 1
├── 0002.pcd => frame 2
...
└── xxxx.pcd => frame n
```

Create a new task with pre-defined annotations. (Class should be configured on your project in advance)

```python
task_id = client.create_sequential_pcd_task(
    project="YOUR_PROJECT_SLUG",
    name="drive_record",
    folder_path="./drive_record/",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    annotations=[
        {
            "type": "cuboid", # annotation class type
            "value": "human", # annotation class value
            "points": {
                "1": { # number of frame
                    "value": [ # For cuboid, it is a 9-digit number.
                        1, # Coordinate X
                        2, # Coordinate Y
                        3, # Coordinate Z
                        1, # Rotation x
                        1, # Rotation Y
                        1, # Rotation Z
                        2, # Length X
                        2, # Length Y
                        2  # Length Z
                    ],
                    # Make sure to set `autogenerated` False for the first and last frame. "1" and "3" frames in this case.
                    # Otherwise, annotation is auto-completed for rest of frames when you edit.
                    "autogenerated": False,
                },
                "2": {
                    "value": [
                        11,
                        12,
                        13,
                        11,
                        11,
                        11,
                        12,
                        12,
                        12
                    ],
                    "autogenerated": True,
                },
                "3": {
                    "value": [
                        21,
                        22,
                        23,
                        21,
                        21,
                        21,
                        22,
                        22,
                        22
                    ],
                    "autogenerated": False,
                },
            },
        },
    ]
)
```

##### Limitation

You can upload up to a size of 30 MB per file.

#### Find Tasks

Find a single task.

```python
task = client.find_sequential_pcd_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
task = client.find_sequential_pcd_task(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 10 tasks)

```python
tasks = client.get_sequential_pcd_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a single task.

```python
task_id = client.update_sequential_pcd_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    annotations=[
        {
            "type": "cuboid",
            "value": "human",
            "points": {
                "1": {
                    "value": [
                        1,
                        2,
                        3,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2
                    ],
                    "autogenerated": False,
                },
                "2": {
                    "value": [
                        11,
                        12,
                        13,
                        11,
                        11,
                        11,
                        12,
                        12,
                        12
                    ],
                    "autogenerated": False,
                },
            },
        },
    ]
)
```

#### Response

Example of a single Sequential PCD task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "YOUR_TASK_NAME",
    "status": "registered",
    "externalStatus": "registered",
    "priority": 10,
    "annotations": [
        {
            "id": "YOUR_TASK_ANNOTATION_ID",
            "type": "cuboid",
            "title": "human",
            "value": "human",
            "color": "#4bdd62",
            "attributes": [],
            "points": {
                "1": {
                    "value": [2.61, 5.07, 0, 0, 0, 0, 2, 2, 2],
                    "autogenerated": False,
                },
                "2": {
                    "value": [2.61, 5.07, 0, 0, 0, 0, 2, 2, 2],
                    "autogenerated": True,
                },
                "3": {
                    "value": [2.61, 5.07, 0, 0, 0, 0, 2, 2, 2],
                    "autogenerated": False,
                },
            },
        },
        {
            "id": "YOUR_TASK_ANNOTATION_ID",
            "type": "cuboid",
            "title": "building",
            "value": "building",
            "color": "#223543",
            "attributes": [],
            "points": {
                "1": {
                    "value": [2.8, -8.64, 0.15, 0, 0, 0, 4.45, 4.2, 2],
                    "autogenerated": False,
                },
                "2": {
                    "value": [2.8, -8.64, 0.15, 0, 0, 0, 4.45, 4.2, 2],
                    "autogenerated": True,
                },
                "3": {
                    "value": [2.8, -8.64, 0.15, 0, 0, 0, 4.45, 4.2, 2],
                    "autogenerated": True,
                },
                "4": {
                    "value": [2.8, -8.64, 0.15, 0, 0, 0, 4.45, 4.2, 2],
                    "autogenerated": True,
                },
                "5": {
                    "value": [2.8, -8.64, 0.15, 0, 0, 0, 4.45, 4.2, 2],
                    "autogenerated": True,
                },
            },
        },
    ],
    "tags": [],
    "assignee": None,
    "reviewer": None,
    "approver": None,
    "externalAssignee": None,
    "externalReviewer": None,
    "externalApprover": None,
    "createdAt": "2023-03-24T08:39:08.524Z",
    "updatedAt": "2023-03-24T08:39:08.524Z",
}
```

### DICOM

Supported following project types:

- DICOM -Bounding Box

#### Create Task

Create a new task.
You should receive task import history status [Find Task Import History](#find-task-import-history).
Once you receive the status completed, you can get the task.

```python
history = client.create_dicom_task(
    project="YOUR_PROJECT_SLUG",
    file_path="./sample.zip"
)
```

#### Limitation

- You can upload up to a size of 2 GB per file.

#### Find Task

Find a single task.

```python
task = client.find_dicom_task(task_id="YOUR_TASK_ID")
```

Find a single task by name.

```python
tasks = client.find_dicom_task_by_name(project="YOUR_PROJECT_SLUG", task_name="YOUR_TASK_NAME")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_dicom_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a single task.

```python
task_id = client.update_dicom_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    assignee="USER_SLUG",
    tags=["tag1", "tag2"]
)
```

#### Response

Example of a single dicom task object

```python
{
    "id": "YOUR_TASK_ID",
    "name": "dicom.zip",
    "url": "YOUR_TASK_URL",
    'height': 512,
    'width': 512,
    "status": "registered",
    "externalStatus": "registered",
    "tags": [],
    "assignee": "ASSIGNEE_NAME",
    "reviewer": "REVIEWER_NAME",
    "externalAssignee": "EXTERNAL_ASSIGNEE_NAME",
    "externalReviewer": "EXTERNAL_REVIEWER_NAME",
    "annotations": [
        {
            "attributes": [],
            "color": "#b36d18",
            "contentId": "CONTENT_ID"
            "points": [100, 200, 100, 200],
            "title": "Heart",
            "type": "bbox",
            "value": "heart"
        }
    ],
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

### Common

APIs for update and delete and count are same over all tasks.

#### Update Task

Update a single task status, tags and assignee.

```python
task_id = client.update_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10, # (optional) none: 0, low: 10, medium: 20, high: 30
    tags=["tag1", "tag2"],
    assignee="USER_SLUG"
)
```

#### Delete Task

Delete a single task.

```python
client.delete_task(task_id="YOUR_TASK_ID")
```

#### Delete Task Annotation

Delete annotations in a task.

```python
client.delete_task_annotations(task_id="YOUR_TASK_ID")
```

#### Get Tasks Id and Name map

```python
id_name_map = client.get_task_id_name_map(project="YOUR_PROJECT_SLUG")
```

#### Count Task

```python
task_count = client.count_tasks(
    project="YOUR_PROJECT_SLUG",
    status="approved", # status can be 'pending', 'registered', 'completed', 'skipped', 'reviewed' 'sent_back', 'approved', 'declined'
    tags=["tag1", "tag2"] # up to 10 tags
)
```

#### Create Task from S3

Task creation from S3.

- Support project

  - Image
  - Video
  - Audio
  - Text

- To use it, you need to set the contents of the following link.
  <https://docs.fastlabel.ai/docs/integrations-aws-s3>

- Setup AWS S3 properties

```python
status = client.update_aws_s3_storage(
    project="YOUR_PROJECT_SLUG",
    bucket_name="S3_BUCKET_NAME",
    bucket_region="S3_REGIONS",
)
```

- Run create task from AWS S3

```python
history = client.create_task_from_aws_s3(
    project="YOUR_PROJECT_SLUG",
)
```

- Get AWS S3 import status

```python
history = client.get_aws_s3_import_status_by_project(
    project="YOUR_PROJECT_SLUG",
)
```

#### Find Task Import History

Find a single history.

```python
history = client.find_history(history_id="YOUR_HISTORY_ID")
```

#### Get Task Import Histories

```python
histories = client.get_histories(project="YOUR_PROJECT_SLUG")
```

#### Response

Example of a single history object

```python
{
    "id": "YOUR_HISTORY_ID",
    "storageType": "zip",
    "status": "running",
    "createdAt": "2021-02-22T11:25:27.158Z",
    "updatedAt": "2021-02-22T11:25:27.158Z"
}
```

## Annotation

### Create Annotation

Create a new annotation.

```python
annotation_id = client.create_annotation(
    project="YOUR_PROJECT_SLUG", type="bbox", value="cat", title="Cat")
```

Create a new annotation with color and attributes.

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

Create a new classification annotation.

```python
annotation_id = client.create_classification_annotation(
    project="YOUR_PROJECT_SLUG", attributes=attributes)
```

### Find Annotation

Find an annotation.

```python
annotation = client.find_annotation(annotation_id="YOUR_ANNOTATION_ID")
```

Find an annotation by value.

```python
annotation = client.find_annotation_by_value(project="YOUR_PROJECT_SLUG", value="cat")
```

Find an annotation by value in classification project.

```python
annotation = client.find_annotation_by_value(
    project="YOUR_PROJECT_SLUG", value="classification") # "classification" is fixed value
```

### Get Annotations

Get annotations. (Up to 1000 annotations)

```python
annotations = client.get_annotations(project="YOUR_PROJECT_SLUG")
```

### Response

Example of an annotation object

```python
{
    "id": "YOUR_ANNOTATION_ID",
    "type": "bbox",
    "value": "cat",
    "title": "Cat",
    "color": "#FF0000",
    "order": 1,
    "vertex": 0,
    "attributes": [
        {
            "id": "YOUR_ATTRIBUTE_ID",
            "key": "kind",
            "name": "Kind",
            "options": [],
            "order": 1,
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
            "order": 2,
            "type": "select",
            "value": ""
        }
    ],
    "createdAt": "2021-05-25T05:36:50.459Z",
    "updatedAt": "2021-05-25T05:36:50.459Z"
}
```

Example when the annotation type is Pose Estimation

```python
{
   "id":"b12c81c3-ddec-4f98-b41b-cef7f77d26a4",
   "type":"pose_estimation",
   "title":"jesture",
   "value":"jesture",
   "color":"#10c414",
   "order":1,
   "attributes": [],
   "keypoints":[
      {
         "id":"b03ea998-a2f1-4733-b7e9-78cdf73bd38a",
         "name":"頭",
         "key":"head",
         "color":"#0033CC",
         "edges":[
            "195f5852-c516-498b-b392-24513ce3ea67",
            "06b5c968-1786-4d75-a719-951e915e5557"
         ],
         "value": []
      },
      {
         "id":"195f5852-c516-498b-b392-24513ce3ea67",
         "name":"右肩",
         "key":"right_shoulder",
         "color":"#0033CC",
         "edges":[
            "b03ea998-a2f1-4733-b7e9-78cdf73bd38a"
         ],
         "value": []
      },
      {
         "id":"06b5c968-1786-4d75-a719-951e915e5557",
         "name":"左肩",
         "key":"left_shoulder",
         "color":"#0033CC",
         "edges":[
            "b03ea998-a2f1-4733-b7e9-78cdf73bd38a"
         ],
         "value": []
      }
   ],
   "createdAt":"2021-11-21T09:59:46.714Z",
   "updatedAt":"2021-11-21T09:59:46.714Z"
}
```

### Update Annotation

Update an annotation.

```python
annotation_id = client.update_annotation(
    annotation_id="YOUR_ANNOTATION_ID", value="cat2", title="Cat2", color="#FF0000")
```

Update an annotation with attributes.

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

Update a classification annotation.

```python
annotation_id = client.update_classification_annotation(
    project="YOUR_PROJECT_SLUG", attributes=attributes)
```

### Delete Annotation

Delete an annotation.

```python
client.delete_annotation(annotation_id="YOUR_ANNOTATION_ID")
```

## Project

### Create Project

Create a new project.

```python
project_id = client.create_project(
    type="image_bbox", name="ImageNet", slug="image-net")
```

### Find Project

Find a project.

```python
project = client.find_project(project_id="YOUR_PROJECT_ID")
```

Find a project by slug.

```python
project = client.find_project_by_slug(slug="YOUR_PROJECT_SLUG")
```

### Get Projects

Get projects. (Up to 1000 projects)

```python
projects = client.get_projects()
```

### Response

Example of a project object

```python
{
    "id": "YOUR_PROJECT_ID",
    "type": "image_bbox",
    "slug": "YOUR_PROJECT_SLUG",
    "name": "YOUR_PROJECT_NAME",
    "isPixel": False,
    "jobSize": 10,
    "status": "active",
    "createdAt": "2021-04-20T03:20:41.427Z",
    "updatedAt": "2021-04-20T03:20:41.427Z",
}
```

### Update Project

Update a project.

```python
project_id = client.update_project(
    project_id="YOUR_PROJECT_ID", name="NewImageNet", slug="new-image-net", job_size=20)
```

### Delete Project

Delete a project.

```python
client.delete_project(project_id="YOUR_PROJECT_ID")
```

### Copy Project

Copy a project.

```python
project_id = client.copy_project(project_id="YOUR_PROJECT_ID")
```

## Tags

### Get Tags

Get tags. (Up to 1000 tags)

keyword are search terms in the tag name (Optional).
offset is the starting position number to fetch (Optional).
limit is the max number to fetch (Optional).

If you need to fetch more than 1000 tags, please loop this method using offset and limit.
In the sample code below, you can fetch 1000 tags starting from the 2001st position.

```python
projects = client.get_tags(
    project="YOUR_PROJECT_SLUG",
    keyword="dog", # (Optional)
    offset=2000,  # (Optional)
    limit=1000,  # (Optional. Default is 100.)
)
```

### Response

Example of tags object

```python
[
    {
        "id": "YOUR_TAG_ID",
        "name": "YOUR_TAG_NAME",
        "order": 1,
        "createdAt": "2023-08-14T11: 32: 36.462Z",
        "updatedAt": "2023-08-14T11: 32: 36.462Z"
    }
]
```

### Delete Tags

Delete tags.

```python
client.delete_tags(
    tag_ids=[
        "YOUR_TAG_ID_1",
        "YOUR_TAG_ID_2",
    ],
)
```

## Dataset

### Create Dataset

Create a new dataset.

```python
dataset = client.create_dataset(
    name="object-detection", # Only lowercase alphanumeric characters + hyphen is available
    tags=["cat", "dog"], # max 5 tags per dataset.
    visibility="workspace", # visibility can be 'workspace' or 'public' or 'organization'
)
```

#### Response Dataset

See API docs for details.

```python
{
    'id': 'YOUR_DATASET_ID',
    'name': 'object-detection',
    'tags': ['cat', 'dog'],
    'visibility': 'workspace',
    'license': 'The MIT License',
    'createdAt': '2022-10-31T02:20:00.248Z',
    'updatedAt': '2022-10-31T02:20:00.248Z'
}
```

### Find Dataset

Find a single dataset.

```python
dataset = client.find_dataset(dataset_id="YOUR_DATASET_ID")
```

Success response is the same as when created.

### Get Dataset

Get all datasets in the workspace. (Up to 1000 tasks)

```python
datasets = client.get_datasets()
```

The success response is the same as when created, but it is an array.

You can filter by keywords and visibility, tags.

```python
datasets = client.get_datasets(
    keyword="dog",
    tags=["cat", "dog"], # max 5 tags per dataset.
    visibility="workspace", # visibility can be 'workspace' or 'public' or 'organization'.
)
```

If you wish to retrieve more than 1000 datasets, please refer to the Task [sample code](#get-tasks).

### Update Dataset

Update a single dataset.

```python
dataset = client.update_dataset(
    dataset_id="YOUR_DATASET_ID", name="object-detection", tags=["cat", "dog"]
)
```

Success response is the same as when created.

### Delete Dataset

Delete a single dataset.

**⚠️ The dataset object and its associated tasks that dataset has will also be deleted, so check carefully before executing.**

```python
client.delete_dataset(dataset_id="YOUR_DATASET_ID")
```

### Create Dataset Object

Create object in the dataset.

The types of objects that can be created are "image", "video", and "audio".
There are type-specific methods. but they can be used in the same way.

Created object are automatically assigned to the "latest" dataset version.

```python
dataset_object = client.create_dataset_object(
    dataset="YOUR_DATASET_NAME",
    name="brushwood_dog.jpg",
    file_path="./brushwood_dog.jpg",
    tags=["dog"], # max 5 tags per dataset object.
    licenses=["MIT", "my-license"],  # max 10 licenses per dataset object
    annotations=[
        {
            "keypoints": [
                {
                    "value": [
                        102.59,
                        23.04,
                        1
                    ],
                    "key": "head"
                }
            ],
            "attributes": [
                {
                    "type": "text",
                    "value": "Scottish field",
                    "key": "kind"
                }
            ],
            "confidenceScore": 0,
            "rotation": 0,
            "points": [
                0
            ],
            "value": "dog",
            "type": "bbox" # type can be 'bbox', 'segmentation'.
        }
    ],
    custom_metadata={
      "key": "value",
      "metadata": "metadata-value"
    }
)
```

If you would like to create a new dataset object with classification type annotations, please pass empty points and value of the annotation named 'classification'.

```python
dataset_object = client.create_dataset_object(
    dataset="YOUR_DATASET_NAME",
    name="brushwood_dog.jpg",
    file_path="./brushwood_dog.jpg",
    tags=["dog"], # max 5 tags per dataset object.
    licenses=["MIT", "my-license"],  # max 10 licenses per dataset object
    annotations=[
        { 
            "type": "classification",
            "value": "classification",
            "points": [],
            "attributes": [
                {
                    "type": "text",
                    "value": "Scottish field",
                    "key": "kind"
                }
            ]
        }
    ]
)
```

#### Response Dataset Object

See API docs for details.

```python
{
    'name': 'brushwood_dog.jpg',
    'size': 6717,
    'height': 225,
    'width': 225,
    'tags': [
        'dog'
    ],
    "annotations": [
        {
            "id": "YOUR_DATASET_OBJECT_ANNOTATION_ID",
            "type": "bbox",
            "title": "dog",
            "value": "dog",
            "points": [
                0
            ],
            "attributes": [
                {
                    "value": "Scottish field",
                    "key": "kind",
                    "name": "Kind",
                    "type": "text"
                }
            ],
            "keypoints": [
                {
                    "edges": [
                        "right_shoulder",
                        "left_shoulder"
                    ],
                    "value": [
                        102.59,
                        23.04,
                        1
                    ],
                    "key": "head",
                    "name": "頭"
                }
            ],
            "rotation": 0,
            "color": "#FF0000",
            "confidenceScore": -1
        }
    ],
  "customMetadata": {
    "key": "value",
    "metadata": "metadata-value"
  },
    'createdAt': '2022-10-30T08:32:20.748Z',
    'updatedAt': '2022-10-30T08:32:20.748Z'
}
```

### Find Dataset Object

Find a single dataset object.

```python
dataset_object = client.find_dataset_object(
    dataset_id="YOUR_DATASET_ID",
    object_name="brushwood_dog.jpg"
)
```

You can find a object of specified revision by version or revision_id.

```python
dataset_object = client.find_dataset_object(
    dataset_id="YOUR_DATASET_ID",
    object_name="brushwood_dog.jpg",
    version="YOUR_VERSION_NAME" # default is "latest"
)
```

```python
dataset_object = client.find_dataset_object(
    dataset_id="YOUR_DATASET_ID",
    object_name="brushwood_dog.jpg",
    revision_id="YOUR_REVISION_ID" # 8 characters or more
)
```

Success response is the same as when created.

### Get Dataset Object

Get all dataset object in the dataset. (Up to 1000 tasks)

```python
dataset_objects = client.get_dataset_objects(dataset="YOUR_DATASET_NAME")
```

The success response is the same as when created, but it is an array.

You can filter by version or revision_id, licenses and tags.

```python
dataset_objects = client.get_dataset_objects(
    dataset="YOUR_DATASET_NAME",
    version="latest", # default is "latest"
    tags=["cat"],
    licenses=["fastlabel"],
    types=["train", "valid"],  # choices are "train", "valid", "test" and "none" (Optional)
    offset=0,  # default is 0 (Optional)
    limit=1000,  # default is 1000, and must be less than 1000 (Optional)
)
```

```python
dataset_objects = client.get_dataset_objects(
    dataset="YOUR_DATASET_NAME",
    revision_id="YOUR_REVISION_ID" # 8 characters or more
)
```

### Download Dataset Objects

Download dataset objects in the dataset to specific directories.

You can filter by version, tags and types.

```python
client.download_dataset_objects(
  dataset="YOUR_DATASET_NAME",
  path="YOUR_DOWNLOAD_PATH",
  version="latest", # default is "latest"
  tags=["cat"],
  types=["train", "valid"],  # choices are "train", "valid", "test" and "none" (Optional)
  licenses=["fastlabel"],
  offset=0,  # default is 0 (Optional)
  limit=1000,  # default is 1000, and must be less than 1000 (Optional)
)
```

### Update Dataset Object
```python
dataset_object = client.update_dataset_object(
    dataset_id="YOUR_DATASET_ID",
    object_name="brushwood_dog.jpg",
    tags=["dog"], # max 5 tags per dataset object.
    licenses=["MIT", "my-license"],  # max 10 licenses per dataset object
    annotations=[
        {
            "keypoints": [
                {
                    "value": [
                        102.59,
                        23.04,
                        1
                    ],
                    "key": "head"
                }
            ],
            "attributes": [
                {
                    "value": "Scottish field",
                    "key": "kind"
                }
            ],
            "confidenceScore": 0,
            "rotation": 0,
            "points": [
                0
            ],
            "value": "dog",
            "type": "bbox" # type can be 'bbox', 'segmentation'.
        }
    ],
    custom_metadata={
      "key": "value",
      "metadata": "metadata-value"
    }
)
```

### Delete Dataset Object

Delete a single dataset object.

```python
client.delete_dataset_object(
    dataset_id="YOUR_DATASET_ID",
    object_name="brushwood_dog.jpg"
)
```

## Converter

### FastLabel To COCO

Support the following annotation types.

- bbox
- polygon
- pose estimation

Get tasks and export as a [COCO format](https://cocodataset.org/#format-data) file.

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
client.export_coco(project=project_slug, tasks=tasks)
```

Export with specifying output directory and file name.

```python
client.export_coco(project="YOUR_PROJECT_SLUG", tasks=tasks, output_dir="YOUR_DIRECTROY", output_file_name="YOUR_FILE_NAME")
```

If you would like to export pose estimation type annotations, please pass annotations.

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
annotations = client.get_annotations(project=project_slug)
client.export_coco(project=project_slug, tasks=tasks, annotations=annotations, output_dir="YOUR_DIRECTROY", output_file_name="YOUR_FILE_NAME")
```

### FastLabel To YOLO

Support the following annotation types.

- bbox
- polygon

Get tasks and export as YOLO format files.

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
client.export_yolo(project=project_slug, tasks=tasks, output_dir="YOUR_DIRECTROY")
```

Get tasks and export as YOLO format files with classes.txt
You can use fixed classes.txt and arrange order of each annotaiton file's order

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
annotations = client.get_annotations(project=project_slug)
classes = list(map(lambda annotation: annotation["value"], annotations))
client.export_yolo(project=project_slug, tasks=tasks, classes=classes, output_dir="YOUR_DIRECTROY")
```

### FastLabel To Pascal VOC

Support the following annotation types.

- bbox
- polygon

Get tasks and export as Pascal VOC format files.

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
client.export_pascalvoc(project=project_slug, tasks=tasks)
```

### FastLabel To labelme

Support the following annotation types.

- bbox
- polygon
- points
- line

Get tasks and export as labelme format files.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_labelme(tasks)
```

### FastLabel To Segmentation

Get tasks and export index color instance/semantic segmentation (PNG files).
Only support the following annotation types.

- bbox
- polygon
- segmentation (Hollowed points are not supported.)

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_instance_segmentation(tasks)
```

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_semantic_segmentation(tasks)
```

### COCO To FastLabel

Supported bbox , polygon or pose_estimation annotation type.

Convert annotation file of [COCO format](https://cocodataset.org/#format-data) as a Fastlabel format and create task.

file_path: COCO annotation json file path

```python
annotations_map = client.convert_coco_to_fastlabel(file_path="./sample.json", annotation_type="bbox")
# annotation_type = "bbox", "polygon" or "pose_estimation

task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    annotations=annotations_map.get("sample.jpg")
)
```

Example of converting annotations to create multiple tasks.

In the case of the following tree structure.

```
dataset
├── annotation.json
├── sample1.jpg
└── sample2.jpg
```

Example source code.

```python
import fastlabel

project = "YOUR_PROJECT_SLUG"
input_file_path = "./dataset/annotation.json"
input_dataset_path = "./dataset/"

annotations_map = client.convert_coco_to_fastlabel(file_path=input_file_path)
for image_file_path in glob.iglob(os.path.join(input_dataset_path, "**/**.jpg"), recursive=True):
    time.sleep(1)
    name = image_file_path.replace(os.path.join(*[input_dataset_path, ""]), "")
    file_path = image_file_path
    annotations = annotations_map.get(name) if annotations_map.get(name) is not None else []
    task_id = client.create_image_task(
        project=project,
        name=name,
        file_path=file_path,
        annotations=annotations
    )
```

### YOLO To FastLabel

Supported bbox annotation type.

Convert annotation file of YOLO format as a Fastlabel format and create task.

classes_file_path: YOLO classes text file path
dataset_folder_path: Folder path containing YOLO Images and annotation

```python
annotations_map = client.convert_yolo_to_fastlabel(
    classes_file_path="./classes.txt",
    dataset_folder_path="./dataset/"
)
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./dataset/sample.jpg",
    annotations=annotations_map.get("sample.jpg")
)
```

Example of converting annotations to create multiple tasks.

In the case of the following tree structure.

```
yolo
├── classes.txt
└── dataset
    ├── sample1.jpg
    ├── sample1.txt
    ├── sample2.jpg
    └── sample2.txt
```

Example source code.

```python
import fastlabel

project = "YOUR_PROJECT_SLUG"
input_file_path = "./classes.txt"
input_dataset_path = "./dataset/"
annotations_map = client.convert_yolo_to_fastlabel(
    classes_file_path=input_file_path,
    dataset_folder_path=input_dataset_path
)
for image_file_path in glob.iglob(os.path.join(input_dataset_path, "**/**.jpg"), recursive=True):
    time.sleep(1)
    name = image_file_path.replace(os.path.join(*[input_dataset_path, ""]), "")
    file_path = image_file_path
    annotations = annotations_map.get(name) if annotations_map.get(name) is not None else []
    task_id = client.create_image_task(
        project=project,
        name=name,
        file_path=file_path,
        annotations=annotations
    )
```

### Pascal VOC To FastLabel

Supported bbox annotation type.

Convert annotation file of Pascal VOC format as a Fastlabel format and create task.

folder_path: Folder path including pascal VOC format annotation files

```python
annotations_map = client.convert_pascalvoc_to_fastlabel(folder_path="./dataset/")
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./dataset/sample.jpg",
    annotations=annotations_map.get("sample.jpg")
)
```

Example of converting annotations to create multiple tasks.

In the case of the following tree structure.

```
dataset
├── sample1.jpg
├── sample1.xml
├── sample2.jpg
└── sample2.xml
```

Example source code.

```python
import fastlabel

project = "YOUR_PROJECT_SLUG"
input_dataset_path = "./dataset/"

annotations_map = client.convert_pascalvoc_to_fastlabel(folder_path=input_dataset_path)
for image_file_path in glob.iglob(os.path.join(input_dataset_path, "**/**.jpg"), recursive=True):
    time.sleep(1)
    name = image_file_path.replace(os.path.join(*[input_dataset_path, ""]), "")
    file_path = image_file_path
    annotations = annotations_map.get(name) if annotations_map.get(name) is not None else []
    task_id = client.create_image_task(
        project=project,
        name=name,
        file_path=file_path,
        annotations=annotations
    )
```

### labelme To FastLabel

Support the following annotation types.

- bbox
- polygon
- points
- line

Convert annotation file of labelme format as a Fastlabel format and create task.

folder_path: Folder path including labelme format annotation files

```python
annotations_map = client.convert_labelme_to_fastlabel(folder_path="./dataset/")
task_id = client.create_image_task(
    project="YOUR_PROJECT_SLUG",
    name="sample.jpg",
    file_path="./sample.jpg",
    annotations=annotations_map.get("sample.jpg")
)
```

Example of converting annotations to create multiple tasks.

In the case of the following tree structure.

```
dataset
├── sample1.jpg
├── sample1.json
├── sample2.jpg
└── sample2.json
```

Example source code.

```python
import fastlabel

project = "YOUR_PROJECT_SLUG"
input_dataset_path = "./dataset/"

annotations_map = client.convert_labelme_to_fastlabel(folder_path=input_dataset_path)
for image_file_path in glob.iglob(os.path.join(input_dataset_path, "**/**.jpg"), recursive=True):
    time.sleep(1)
    name = image_file_path.replace(os.path.join(*[input_dataset_path, ""]), "")
    file_path = image_file_path
    annotations = annotations_map.get(name) if annotations_map.get(name) is not None else []
    task_id = client.create_image_task(
        project=project,
        name=name,
        file_path=file_path,
        annotations=annotations
    )
```

> Please check const.COLOR_PALLETE for index colors.

## Model

### Get training jobs

Get training jobs.

```python
def get_training_jobs() -> list[dict]:
    all_training_jobs = []
    offset = None
    while True:
        time.sleep(1)

        training_jobs = client.get_training_jobs(offset=offset)
        all_training_jobs.extend(training_jobs)

        if len(training_jobs) > 0:
            offset = len(all_training_jobs)
        else:
            break
    return all_training_jobs

```

#### Find Training job

Find a single training job.

```python
task = client.find_training_job(id="YOUR_TRAINING_ID")
```

#### Response

Example of two training jobs.

```python
[
    {
        "trainingJobId": "f40c5838-4c3a-482f-96b7-f77e16c96fed",
        "status": "in_progress",
        "baseModelName": "FastLabel Object Detection High Accuracy - 汎用",
        "instanceType": "ml.p3.2xlarge",
        "epoch": 300,
        "projects": [
            "image-bbox"
        ],
        "statuses": [],
        "tags": [],
        "contentCount": 23,
        "userName": "Admin",
        "createdAt": "2023-10-31T07:10:28.306Z",
        "completedAt": null,
        "customModel": {
            "modelId": "",
            "modelName": "",
            "modelURL": "",
            "classes": []
        }
    },
    {
        "trainingJobId": "1d2bc86a-c7f1-40a5-8e85-48246cc3c8d2",
        "status": "completed",
        "baseModelName": "custom-object-detection-image",
        "instanceType": "ml.p3.2xlarge",
        "epoch": 300,
        "projects": [
            "image-bbox"
        ],
        "statuses": [
            "approved"
        ],
        "tags": [
            "trainval"
        ],
        "contentCount": 20,
        "userName": "Admin",
        "createdAt": "2023-10-31T06:56:28.112Z",
        "completedAt": "2023-10-31T07:08:26.000Z",
        "customModel": {
            "modelId": "a6728876-2eb7-49b5-9fd8-7dee1b8a81b3",
            "modelName": "fastlabel_object_detection-2023-10-31-07-08-29",
            "modelURL": "URL for download model file",
            "classes": [
                "person"
            ]
        }
    }
]
```

### Execute training job

Get training jobs.

```python
training_job = client.execute_training_job(
    dataset_name="dataset_name",
    base_model_name="fastlabel_object_detection_light",  // "fastlabel_object_detection_light" or "fastlabel_object_detection_high_accuracy" or "fastlabel_u_net_general"
    epoch=300,
    use_dataset_train_val=True,
    resize_option="fixed",  // optional, "fixed" or "none"
    resize_dimension=1024, // optional, 512 or 1024
    annotation_value="person", // Annotation value is required if choose "fastlabel_keypoint_rcnn"
    config_file_path="config.yaml", // optional, YAML file path for training config file.
)

```

### Get evaluation jobs

Get evaluation jobs.

```python
def get_evaluation_jobs() -> list[dict]:
    all_evaluation_jobs = []
    offset = None
    while True:
        time.sleep(1)

        evaluation_jobs = client.get_evaluation_jobs(offset=offset)
        all_evaluation_jobs.extend(evaluation_jobs)

        if len(evaluation_jobs) > 0:
            offset = len(all_evaluation_jobs)
        else:
            break
    return all_evaluation_jobs

```

#### Find Evaluation job

Find a single evaluation job.

```python
evaluation_job = client.find_evaluation_job(id="YOUR_EVALUATION_ID")
```

#### Response

Example of two evaluation jobs.

```python

{
  id: "50873ea1-e008-48db-a368-241ca88d6f67",
  version: 59,
  status: "in_progress",
  modelType: "builtin",
  modelName: "FastLabel Object Detection Light - 汎用",
  customModelId: None,
  iouThreshold: 0.8,
  confidenceThreshold: 0.4,
  contentCount: 0,
  gtCount: 0,
  predCount: 0,
  mAP: 0,
  recall: 0,
  precision: 0,
  f1: 0,
  confusionMatrix: None,
  duration: 0,
  evaluationSource: "dataset",
  projects: [],
  statuses: [],
  tags: [],
  datasetId: "deacbe6d-406f-4086-bd87-80ffb1c1a393",
  dataset: {
    id: "deacbe6d-406f-4086-bd87-80ffb1c1a393",
    workspaceId: "df201d3c-af00-423a-aa7f-827376fd96de",
    name: "sample-dataset",
    createdAt: "2023-12-20T10:44:12.198Z",
    updatedAt: "2023-12-20T10:44:12.198Z",
  },
  datasetRevisionId: "2d26ab64-dfc0-482d-9211-ce8feb3d480b",
  useDatasetTest: True,
  userName: "",
  completedAt: None,
  createdAt: "2023-12-21T09:08:16.111Z",
  updatedAt: "2023-12-21T09:08:18.414Z",
};

```

### Execute evaluation job

Execute evaluation jobs.

```python
training_job = client.execute_evaluation_job(
    dataset_name="DATASET_NAME",
    model_name="fastlabel_object_detection_light",
    // If you want to use the built-in model, select the following.
    - "fastlabel_object_detection_light"
    - "fastlabel_object_detection_high_accuracy"
    - "fastlabel_fcn_resnet"
    // If you want to use the custom model, please fill　out model name.
    use_dataset_test=True,
)

```

### Execute endpoint

Create the endpoint from the screen at first.

Currently, the feature to create endpoints is in alpha and is not available to users.
If you would like to try it out, please contact a FastLabel representative.

```python
import fastlabel
import numpy as np
import cv2
import base64
client = fastlabel.Client()

ENDPOINT_NAME = "YOUR ENDPOINT NAME"
IMAGE_FILE_PATH = "YOUR IMAGE FILE PATH"
RESULT_IMAGE_FILE_PATH = "YOUR RESULT IMAGE FILE PATH"

def base64_to_cv(img_str):
  if "base64," in img_str:
      img_str = img_str.split(",")[1]
  img_raw = np.frombuffer(base64.b64decode(img_str), np.uint8)
  img = cv2.imdecode(img_raw, cv2.IMREAD_UNCHANGED)
  return img

if __name__ == '__main__':
  # Execute endpoint
  response = client.execute_endpoint(
      endpoint_name=ENDPOINT_NAME, file_path=IMAGE_PATH)

  # Show  result
  print(response["json"])

  # Save result
  img = base64_to_cv(response["file"])
  cv2.imwrite(RESULT_IMAGE_FILE_PATH, img)
```

### Create Request Results for Monitoring

You can integrate the results of model endpoint calls,
which are targeted for aggregation in model monitoring, from an external source.

```python
from datetime import datetime
import pytz
import fastlabel
client = fastlabel.Client()

jst = pytz.timezone("Asia/Tokyo")
dt_jst = datetime(2023, 5, 8, 12, 10, 53, tzinfo=jst)

client.create_model_monitoring_request_results(
    name="model-monitoring-name",  # The name of your model monitoring name
    results=[
        {
            "status": "success",  # success or failed
            "result": [
                {
                    "value": "person",  # The value of the inference class returned by your model
                    "confidenceScore": 0.92,  # 0 ~ 1
                }
            ],
            "requestAt": dt_jst.isoformat(),  # The time when your endpoint accepted the request
        }
    ],
)
```

## API Docs

Check [this](https://api.fastlabel.ai/docs/) for further information.
