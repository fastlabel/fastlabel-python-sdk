from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object = client.find_dataset_object(
    dataset_id="YOUR_DATASET_OBJECT_ID", object_name="YOUR_OBJECT_NAME"
)
pprint(dataset_object)
