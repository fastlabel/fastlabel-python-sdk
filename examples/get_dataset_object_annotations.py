from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object_annotations = client.get_dataset_object_annotations(
    dataset_object_id="YOUR_DATASET_OBJECT_ID",
)
pprint(dataset_object_annotations)
