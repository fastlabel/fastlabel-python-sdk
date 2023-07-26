from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object = client.create_image_dataset_object(
    dataset_version_id="YOUR_DATASET_VERSION_ID",
    name="NAME",
    file_path="FILE_PATH",
)
pprint(dataset_object)
