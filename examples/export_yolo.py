import fastlabel
client = fastlabel.Client()

project_slug = "asweer"
tasks = client.get_image_tasks(project=project_slug)

# client.export_yolo(project=project_slug, tasks=tasks, output_dir="./export/")

client.export_yolo(project=project_slug, classes=["dog"], tasks=tasks, output_dir="./export/")