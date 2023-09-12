from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object = client.create_dataset_object(
    dataset="YOUR_DATASET_NAME",
    name="NAME",
    file_path="FILE_PATH",
)
pprint(dataset_object)
