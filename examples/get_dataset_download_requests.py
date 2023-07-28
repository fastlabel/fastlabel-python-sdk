from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_download_requests = client.get_dataset_download_requests()
pprint(dataset_download_requests)
