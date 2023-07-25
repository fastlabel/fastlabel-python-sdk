from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_objects = client.get_dataset_objects(
    dataset_version_id="YOUR_DATASET_VERSION_ID"
)
pprint(dataset_objects)
