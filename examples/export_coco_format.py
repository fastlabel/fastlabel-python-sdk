"""
Description:
This script exports FastLabel annotations in coco format.

Usage:
1. prepare a project with registered tasks and annotations
2. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
3. set PROJECT_SLUG to the project slug you prepared
4. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
PROJECT_SLUG = "fastlabel-sample" # replace with your project slug

import fastlabel
import os

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

# export coco format
tasks = client.get_image_tasks(project=PROJECT_SLUG)
client.export_coco(project=PROJECT_SLUG, tasks=tasks)