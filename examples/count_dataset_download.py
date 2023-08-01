from pprint import pprint

import fastlabel

client = fastlabel.Client()

count = client.count_dataset_download()
pprint(count)
