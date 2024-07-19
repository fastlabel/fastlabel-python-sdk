import fastlabel

client = fastlabel.Client()

client.download_dataset_objects(
    dataset="object-detection", path="./downloads", types=["train", "valid"]
)
