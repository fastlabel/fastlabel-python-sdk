"""
Description:
This script get all tasks registered in the project.

Usage:
1. prepare a project with registered tasks
2. set environment variable FASTLABEL_ACCESS_TOKEN to your access token
3. set PROJECT_SLUG to the project slug you prepared
3. run the script
"""
FASTLABEL_ACCESS_TOKEN = "sample-token" # replace with your access token
PROJECT_SLUG = "fastlabel-sample" # replace with your project slug

import fastlabel
import os

# Initialize client
os.environ['FASTLABEL_ACCESS_TOKEN'] = FASTLABEL_ACCESS_TOKEN
client = fastlabel.Client()

def get_all_tasks(
    project_slug: str,
    status: str = None,
    external_status: str = None,
    tags: list = []
    ) -> list:

    # Iterate pages until new tasks are empty.
    result = []
    offset = None

    print("Fetching tasks...")
    while True:
        tasks = client.get_image_tasks(
            # Change to another method (e.g., get_video_tasks) based on your requirement.
            project=project_slug,
            status=status,
            external_status=external_status,
            tags=tags,
            limit=100,
            offset=offset
        )
        result.extend(tasks)

        if len(tasks) > 0:
            offset = len(result)
            print(offset, ' fetched')
        else:
            break

    print("Fetching tasks finished!")
    print("Total tasks count: ", len(result))
    return result


if __name__ == "__main__":
    print(get_all_tasks(project_slug=PROJECT_SLUG))