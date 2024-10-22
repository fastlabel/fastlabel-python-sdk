import fastlabel

client = fastlabel.Client()

client.download_dataset_objects(
    dataset="testttt", path="./downloads", types=["train", "valid"]
)
