from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_attributes = client.get_dataset_attributes(
    dataset_version_id="YOUR_DATASET_VERSION_ID",
)
pprint(dataset_attributes)
