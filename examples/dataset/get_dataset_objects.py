from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_objects = client.get_dataset_objects(dataset="YOUR_DATASET_NAME")
pprint(dataset_objects)
