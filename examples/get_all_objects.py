"""
Description:
This script get all objects registered in the dataset.

Usage:
1. prepare a dataset with registered objects
2. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
3. set DATASET_NAME to the dataset name you prepared
3. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
DATASET_NAME = "fastlabel-sample" # replace with your dataset name

import fastlabel
import os

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

def get_all_dataset_objects(dataset_name: str, tags: list[str] = None):
    # Iterate pages until new tasks are empty.
    result = []
    offset = None

    print("Fetching dataset objects...")
    while True:
        datasets = client.get_dataset_objects(dataset=dataset_name, tags=tags, offset=offset)
        result.extend(datasets)

        if len(datasets) > 0:
            offset = len(result)
            print(offset, ' fetched')
        else:
            break

    print("Fetching dataset objects finished!")
    print("Total dataset object count: ", len(result))
    return result

if __name__ == "__main__":
    print(get_all_dataset_objects(dataset_name=DATASET_NAME))