import os
import glob
from logging import getLogger

from .exceptions import FastLabelInvalidException
from .api import Api
from fastlabel import converters, utils

logger = getLogger(__name__)


class Client:

    api = None

    def __init__(self):
        self.api = Api()

    # Task Find

    def find_image_task(self, task_id: str) -> dict:
        """
        Find a signle image task.
        """
        endpoint = "tasks/image/" + task_id
        return self.api.get_request(endpoint)

    def find_image_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a signle image task by name.

        project is slug of your project. (Required)
        task_name is a task name. (Required)
        """
        tasks = self.get_image_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_image_classification_task(self, task_id: str) -> dict:
        """
        Find a signle image classification task.
        """
        endpoint = "tasks/image/classification/" + task_id
        return self.api.get_request(endpoint)

    def find_image_classification_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a signle image classification task by name.

        project is slug of your project. (Required)
        task_name is a task name. (Required)
        """
        tasks = self.get_image_classification_tasks(
            project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_multi_image_task(self, task_id: str) -> dict:
        """
        Find a signle multi image task.
        """
        endpoint = "tasks/multi-image/" + task_id
        return self.api.get_request(endpoint)

    def find_multi_image_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a signle multi image task by name.

        project is slug of your project. (Required)
        task_name is a task name. (Required)
        """
        tasks = self.get_multi_image_tasks(
            project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_video_task(self, task_id: str) -> dict:
        """
        Find a signle video task.
        """
        endpoint = "tasks/video/" + task_id
        return self.api.get_request(endpoint)

    def find_video_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a signle video task by name.

        project is slug of your project. (Required)
        task_name is a task name. (Required)
        """
        tasks = self.get_video_tasks(
            project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    # Task Get

    def get_image_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of image tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag. (Optional)
        task_name is a task name. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422)
        endpoint = "tasks/image"
        params = {"project": project}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        if task_name:
            params["taskName"] = task_name
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_image_classification_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of image classification tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        endpoint = "tasks/image/classification"
        params = {"project": project}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        if task_name:
            params["taskName"] = task_name
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_multi_image_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 10,
    ) -> list:
        """
        Returns a list of multi image tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 10:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 10.", 422)
        endpoint = "tasks/multi-image"
        params = {"project": project}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        if task_name:
            params["taskName"] = task_name
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_video_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 10,
    ) -> list:
        """
        Returns a list of video tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag. (Optional)
        task_name is a task name. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 10:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 10.", 422)
        endpoint = "tasks/video"
        params = {"project": project}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        if task_name:
            params["taskName"] = task_name
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_task_id_name_map(
        self, project: str,
        offset: int = None,
        limit: int = 1000,
    ) -> list:
        """
        Returns a map of task ids and names.
        e.g.) {
                "88e74507-07b5-4607-a130-cb6316ca872c", "01_cat.jpg",
                "fe2c24a4-8270-46eb-9c78-bb7281c8bdgs", "02_cat.jpg"
              }
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422)
        endpoint = "tasks/map/id-name"
        params = {"project": project}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    # Task Create

    def create_image_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        annotations: list = [],
        tags: list = [],
    ) -> str:
        """
        Create a single image task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are png, jpg, jpeg. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        annotations is a list of annotation to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        """
        endpoint = "tasks/image"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422)
        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = annotations
        if tags:
            payload["tags"] = tags
        return self.api.post_request(endpoint, payload=payload)

    def create_image_classification_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        attributes: list = [],
        tags: list = [],
    ) -> str:
        """
        Create a single image classification task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are png, jpg, jpeg. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        attributes is a list of attribute to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        """
        endpoint = "tasks/image/classification"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422)
        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if attributes:
            payload["attributes"] = attributes
        if tags:
            payload["tags"] = tags
        return self.api.post_request(endpoint, payload=payload)

    def create_multi_image_task(
        self,
        project: str,
        name: str,
        folder_path: str,
        status: str = None,
        annotations: list = [],
        tags: list = [],
    ) -> str:
        """
        Create a single multi image task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        folder_path is a path to data folder. Files should be under the folder. Nested folder structure is not supported. Supported extensions of files are png, jpg, jpeg. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        annotations is a list of annotation to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        """
        if not os.path.isdir(folder_path):
            raise FastLabelInvalidException(
                "Folder does not exist.", 422)

        endpoint = "tasks/multi-image"
        file_paths = glob.glob(os.path.join(folder_path, "*"))
        contents = []
        for file_path in file_paths:
            if not utils.is_image_supported_ext(file_path):
                raise FastLabelInvalidException(
                    "Supported extensions are png, jpg, jpeg.", 422)
            file = utils.base64_encode(file_path)
            contents.append({
                "name": os.path.basename(file_path),
                "file": file
            })
        payload = {"project": project, "name": name, "contents": contents}
        if status:
            payload["status"] = status
        if annotations:
            payload["annotations"] = annotations
        if tags:
            payload["tags"] = tags
        return self.api.post_request(endpoint, payload=payload)

    def create_video_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        tags: list = [],
    ) -> str:
        """
        Create a single video task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are png, jpg, jpeg. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        """
        endpoint = "tasks/video"
        if not utils.is_video_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are mp4.", 422)
        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if tags:
            payload["tags"] = tags
        return self.api.post_request(endpoint, payload=payload)

    # Task Update

    def update_task(
        self,
        task_id: str,
        status: str = None,
        tags: list = [],
    ) -> str:
        """
        Update a single task.

        task_id is an id of the task. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag to be set. (Optional)
        """
        endpoint = "tasks/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if tags:
            payload["tags"] = tags
        return self.api.put_request(endpoint, payload=payload)

    # Task Delete

    def delete_task(self, task_id: str) -> None:
        """
        Delete a single task.
        """
        endpoint = "tasks/" + task_id
        self.api.delete_request(endpoint)

    # Task Convert

    def to_coco(self, tasks: list) -> dict:
        """
        Convert tasks to COCO format.
        """

        return converters.to_coco(tasks)

    # Annotation

    def find_annotation(self, annotation_id: str) -> dict:
        """
        Find an annotation.
        """
        endpoint = "annotations/" + annotation_id
        return self.api.get_request(endpoint)

    def find_annotation_by_value(self, project: str, value: str) -> dict:
        """
        Find an annotation by value.
        """
        annotations = self.get_annotations(project=project, value=value)
        if not annotations:
            return None
        return annotations[0]

    def get_annotations(
        self,
        project: str,
        value: str = None,
        offset: int = None,
        limit: int = 10,
    ) -> list:
        """
        Returns a list of annotations.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        value is an unique identifier of annotation in your project. (Required)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422)
        endpoint = "annotations"
        params = {"project": project}
        if value:
            params["value"] = value
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def create_annotation(
        self,
        project: str,
        type: str,
        value: str,
        title: str,
        color: str,
        attributes: list = []
    ) -> str:
        """
        Create an annotation.

        project is slug of your project. (Required)
        type can be 'bbox', 'polygon', 'keypoint', 'classification', 'line', 'segmentation'. (Required)
        value is an unique identifier of annotation in your project. (Required)
        title is a display name of value. (Required)
        color is hex color code like #ffffff. (Required)
        attributes is a list of attribute. (Optional)
        """
        endpoint = "annotations"
        payload = {
            "project": project,
            "type": type,
            "value": value,
            "title": title,
            "color": color
        }
        if attributes:
            payload["attributes"] = attributes
        return self.api.post_request(endpoint, payload=payload)

    def create_classification_annotation(
        self,
        project: str,
        attributes: list
    ) -> str:
        """
        Create a classification annotation.

        project is slug of your project. (Required)
        attributes is a list of attribute. (Required)
        """
        endpoint = "annotations/classification"
        payload = {"project": project, "attributes": attributes}
        return self.api.post_request(endpoint, payload=payload)

    def update_annotation(
        self,
        annotation_id: str,
        value: str = None,
        title: str = None,
        color: str = None,
        attributes: list = []
    ) -> str:
        """
        Update an annotation.

        annotation_id is an id of the annotation. (Required)
        value is an unique identifier of annotation in your project. (Optional)
        title is a display name of value. (Optional)
        color is hex color code like #ffffff. (Optional)
        attributes is a list of attribute. (Optional)
        """
        endpoint = "annotations/" + annotation_id
        payload = {}
        if value:
            payload["value"] = value
        if title:
            payload["title"] = title
        if color:
            payload["color"] = color
        if attributes:
            payload["attributes"] = attributes
        return self.api.put_request(endpoint, payload=payload)

    def update_classification_annotation(
        self,
        annotation_id: str,
        attributes: list
    ) -> str:
        """
        Update a classification annotation.

        annotation_id is an id of the annotation. (Required)
        attributes is a list of attribute. (Required)
        """
        endpoint = "annotations/classification/" + annotation_id
        payload = {"attributes": attributes}
        return self.api.put_request(endpoint, payload=payload)

    def delete_annotation(self, annotation_id: str) -> None:
        """
        Delete an annotation.
        """
        endpoint = "annotations/" + annotation_id
        self.api.delete_request(endpoint)
