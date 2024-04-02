from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object = client.update_dataset_object(
    dataset_id="YOUR_DATASET_ID",
    object_name="OBJECT_NAME",
    tags=["TAG1", "TAG2"],
)
pprint(dataset_object)
