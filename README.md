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
  - [Video Classification](#video-classification)
  - [Common](#common)
- [Annotation](#annotation)
- [Project](#project)
- [Converter](#converter)
  - [COCO](#coco)
  - [YOLO](#yolo)
  - [Pascal VOC](#pascal-voc)
  - [labelme](#labelme)
  - [Segmentation](#segmentation)
- [Converter to FastLabel format](#converter-to-fastlabel-format)

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
- Image - Pose Estimation(not support Create Task)
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

Update a signle task.

```python
task_id = client.update_image_task(
    task_id="YOUR_TASK_ID",
    status="approved",
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
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

#### Find Task

Find a single task.

```python
task = client.find_image_classification_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_image_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a signle task.

```python
task_id = client.update_image_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    assignee="USER_SLUG",
    tags=["tag1", "tag2"]
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

### Multi Image

Supported following project types:

- Multi Image - Bounding Box
- Multi Image - Polygon
- Multi Image - Keypoint
- Multi Image - Line
- Multi Image - Segmentation

#### Create Task

Create a new task.

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

##### Limitation
* You can upload up to a total size of 512 MB.
* You can upload up to 250 files in total.

#### Find Task

Find a single task.

```python
task = client.find_multi_image_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

Get tasks.

```python
tasks = client.get_multi_image_tasks(project="YOUR_PROJECT_SLUG")
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
* You can upload up to a size of 250 MB.

#### Find Task

Find a single task.

```python
task = client.find_video_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

Get tasks. (Up to 10 tasks)

```python
tasks = client.get_video_tasks(project="YOUR_PROJECT_SLUG")
```

#### Response

Example of a single image classification task object

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
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

##### Limitation
* You can upload up to a size of 250 MB.

#### Find Task

Find a single task.

```python
task = client.find_video_classification_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks

Get tasks. (Up to 1000 tasks)

```python
tasks = client.get_video_classification_tasks(project="YOUR_PROJECT_SLUG")
```

#### Update Tasks

Update a signle task.

```python
task_id = client.update_video_classification_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    assignee="USER_SLUG",
    tags=["tag1", "tag2"]
    attributes=[
        {
            "key": "attribute-key",
            "value": "attribute-value"
        }
    ],
)
```

### Common

APIs for update and delete are same over all tasks.

#### Update Task

Update a single task status, tags and assignee.

```python
task_id = client.update_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    tags=["tag1", "tag2"],
    assignee="USER_SLUG"
)
```

#### Delete Task

Delete a single task.

```python
client.delete_task(task_id="YOUR_TASK_ID")
```

#### Get Tasks Id and Name map

```python
id_name_map = client.get_task_id_name_map(project="YOUR_PROJECT_SLUG")
```

## Annotation

### Create Annotaion

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

## Converter

### COCO

Support the following annotation types.

- bbox
- polygon
- pose estimation

Get tasks and export as a [COCO format](https://cocodataset.org/#format-data) file.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_coco(tasks)
```

Export with specifying output directory and file name.

```python
client.export_coco(tasks=tasks, output_dir="YOUR_DIRECTROY", output_file_name="YOUR_FILE_NAME")
```

If you would like to export pose estimation type annotations, please pass annotations.

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
annotations = client.get_annotations(project=project_slug)
client.export_coco(tasks=tasks, annotations=annotations, output_dir="YOUR_DIRECTROY", output_file_name="YOUR_FILE_NAME")
```

### YOLO

Support the following annotation types.

- bbox
- polygon

Get tasks and export as YOLO format files.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_yolo(tasks, output_dir="YOUR_DIRECTROY")
```

Get tasks and export as YOLO format files with classes.txt
You can use fixed classes.txt and arrange order of each annotaiton file's order

```python
project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
annotations = client.get_annotations(project=project_slug)
classes = list(map(lambda annotation: annotation["value"], annotations))
client.export_yolo(tasks=tasks, classes=classes, output_dir="YOUR_DIRECTROY")
```

### Pascal VOC

Support the following annotation types.

- bbox
- polygon

Get tasks and export as Pascal VOC format files.

```python
tasks = client.get_image_tasks(project="YOUR_PROJECT_SLUG")
client.export_pascalvoc(tasks)
```

### labelme

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

### Segmentation

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

## Converter to FastLabel format

### Response

Example of a converted annotations

```python
{
  'sample1.jpg':  [
    {
      'points': [
        100,
        100,
        200,
        200
      ],
      'type': 'bbox',
      'value': 'cat'
    }
  ],
  'sample2.jpg':  [
    {
      'points': [
        100,
        100,
        200,
        200
      ],
      'type': 'bbox',
      'value': 'cat'
    }
  ]
}
```

In the case of YOLO, Pascal VOC, and labelme, the key is the tree structure if the tree structure is multi-level.

```
dataset
├── sample1.jpg
├── sample1.txt
└── sample_dir
    ├── sample2.jpg
    └── sample2.txt
```

```python
{
  'sample1.jpg':  [
    {
      'points': [
        100,
        100,
        200,
        200
      ],
      'type': 'bbox',
      'value': 'cat'
    }
  ],
  'sample_dir/sample2.jpg':  [
    {
      'points': [
        100,
        100,
        200,
        200
      ],
      'type': 'bbox',
      'value': 'cat'
    }
  ]
}
```

### COCO

Supported bbox or polygon annotation type.

Convert annotation file of [COCO format](https://cocodataset.org/#format-data) as a Fastlabel format and create task.

file_path: COCO annotation json file path

```python
annotations_map = client.convert_coco_to_fastlabel(file_path="./sample.json")
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

### YOLO

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

### Pascal VOC

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

### labelme

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

## API Docs

Check [this](https://api.fastlabel.ai/docs/) for further information.
