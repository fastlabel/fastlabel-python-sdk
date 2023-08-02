from pprint import pprint

import fastlabel

client = fastlabel.Client()

stats = client.get_dataset_object_annotation_stats(
    dataset_version_id="YOUR_DATASET_VERSION_ID",
)
pprint(stats)
