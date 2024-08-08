"""
Description:
This script creates a project and creates tasks for each image in the project.

Usage:
1. if you already have a project with the flag PROJECT_FLAG, please delete it
2. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
3. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
PROJECT_NAME = "fastlabel-sample" # don't need to change
PROJECT_SLUG = "fastlabel-sample" # don't need to change

import fastlabel
import os

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

# Create a project
project = client.create_project(type="image_bbox", name=PROJECT_NAME, slug=PROJECT_SLUG) 

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
)