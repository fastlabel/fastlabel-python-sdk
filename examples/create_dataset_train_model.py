"""
Description:
This script creates a dataset from the annotations exported by FastLabel and their images and trains the model.

Usage:
1. if you already have a dataset with the NAME DATASET_NAME, please delete it
2. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
3. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
DATASET_NAME = "fastlabel-sample" # don't need to change

import fastlabel
import os
import json

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

# create dataset
dataset = client.create_dataset(
    name=DATASET_NAME,
)

# load annotations
def remove_unnecessary_keys(annotations):
    keys_to_remove = ['id', 'color']
    clean_annotation = [{k: v for k, v in annotation.items() if k not in keys_to_remove} for annotation in annotations]
    return clean_annotation

with open("./examples/sample_data/annotation/fastlabel/annotations.json", "r") as f:
    annotations = json.load(f)
    name_annotation_map = {}
    for annotation in annotations:
        name = annotation["name"]
        name_annotation_map[name] = remove_unnecessary_keys(annotation["annotations"])

# create dataset objects
objects = [
    {
        "name": "sample1.jpg",
        "file_path": "./examples/sample_data/images/sample1.jpg"
    },
    {
        "name": "sample2.jpg",
        "file_path": "./examples/sample_data/images/sample2.jpg"
    }
]

for object in objects:
    dataset_object = client.create_dataset_object(
        dataset=DATASET_NAME,
        name=object["name"],
        file_path=object["file_path"],
        tags=[], # max 5 tags per dataset object.
        licenses=["MIT"],  # max 10 licenses per dataset object
        annotations=name_annotation_map[object["name"]]
    )

# train model
training_job = client.execute_training_job(
    dataset_name=DATASET_NAME,
    base_model_name="fastlabel_object_detection_light",
    epoch=300,
)