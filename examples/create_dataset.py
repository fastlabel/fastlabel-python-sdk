from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset = client.create_dataset(
    name="Japanese Dogs",
    slug="japanese-dogs2",
    type="video",
    annotation_type="image_bbox",
)
pprint(dataset)
