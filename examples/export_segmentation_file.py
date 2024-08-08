"""
Description:
This script outputs the image annotated by segmentation as an instance/semantic mask image.

Usage:
1. prepare a project with registered tasks and annotations (the annotation type must be segmentation)
1. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
2. set PROJECT_SLUG to the project slug you prepared
3. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
PROJECT_SLUG = "fastlabel-sample" # replace with your project slug

import fastlabel
import os

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

# export segmentation file
tasks = client.get_image_tasks(project=PROJECT_SLUG)
client.export_instance_segmentation(tasks)

tasks = client.get_image_tasks(project=PROJECT_SLUG)
client.export_semantic_segmentation(tasks)