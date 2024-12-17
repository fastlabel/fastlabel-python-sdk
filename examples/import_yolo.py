import glob
import os
import time

import fastlabel

client = fastlabel.Client()

project = "YOUR_PROJECT_SLUG"

input_file_path = "./classes.txt"
input_dataset_path = "./dataset/"
annotations_map = client.convert_yolo_to_fastlabel(
    classes_file_path=input_file_path,
    dataset_folder_path=input_dataset_path,
    project_type="segmentation",
)
for image_file_path in glob.iglob(
    os.path.join(input_dataset_path, "**/**.jpg"), recursive=True
):
    time.sleep(1)
    name = image_file_path.replace(os.path.join(*[input_dataset_path, ""]), "")
    file_path = image_file_path
    annotations = (
        annotations_map.get(name) if annotations_map.get(name) is not None else []
    )
    task_id = client.create_image_task(
        project=project, name=name, file_path=file_path, annotations=annotations
    )
