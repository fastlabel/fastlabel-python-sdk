"""
Description:
This script creates Fastlabel projects, tasks and annotations from coco format data.

Usage:
1. if you already have a project with the flag PROJECT_FLUG, please delete it
2. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
3. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
PROJECT_NAME = "fastlabel-sample" # don't need to change
PROJECT_SLUG = "fastlabel-sample" # don't need to change
ANNOTATION_TYPE = "polygon" # don't need to change

import fastlabel
import os

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

# Create a project
project = client.create_project(type="image_polygon", name=PROJECT_NAME, slug=PROJECT_SLUG) 

# Read annotations from COCO format
annotations_map = client.convert_coco_to_fastlabel(file_path="./examples/sample_data/annotation/coco/annotations.json", annotation_type=ANNOTATION_TYPE)

# Create annotation class
value_set = set()
for key in annotations_map.keys():
    for annotation in annotations_map[key]:
        value_set.add(annotation["value"])
for value in value_set:
    annotation_id = client.create_annotation(project=PROJECT_SLUG, type=ANNOTATION_TYPE, value=value, title=value)

# Create tasks
tasks = [
    {
        "name": "sample1.jpg",
        "file_path": "./examples/sample_data/images/sample1.jpg"
    },
    {
        "name": "sample2.jpg",
        "file_path": "./examples/sample_data/images/sample2.jpg"
    }
]

for task in tasks:
    task_id = client.create_image_task(
    project=PROJECT_SLUG,
    name=task["name"],
    file_path=task["file_path"],
    annotations=annotations_map.get(task["name"])
)