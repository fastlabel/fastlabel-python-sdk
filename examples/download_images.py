import os
import time
import urllib.request
from multiprocessing import Pool, cpu_count

import fastlabel

client = fastlabel.Client()
IMAGE_DIR = "images"
PROJECT_SLUG = "YOUR_PROJECT_SLUG"


def get_all_tasks() -> list:
    # Iterate pages until new tasks are empty.
    all_tasks = []
    offset = None
    while True:
        time.sleep(1)

        tasks = client.get_image_classification_tasks(
            project=PROJECT_SLUG, limit=1000, offset=offset
        )
        all_tasks.extend(tasks)

        if len(tasks) > 0:
            offset = len(all_tasks)  # Set the offset
        else:
            break

    return all_tasks


def download_image(task: dict):
    urllib.request.urlretrieve(task["url"], os.path.join(IMAGE_DIR, task["name"]))


if __name__ == "__main__":
    os.makedirs(IMAGE_DIR, exist_ok=True)

    tasks = get_all_tasks()
    with Pool(cpu_count()) as p:
        p.map(download_image, tasks)
