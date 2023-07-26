from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object_import_histories = client.get_dataset_object_import_histories(
    dataset_version_id="YOUR_DATASET_VERSION_ID"
)
pprint(dataset_object_import_histories)
