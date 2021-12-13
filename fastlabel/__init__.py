import glob
import json
import os
import re
from logging import getLogger
from typing import List

import cv2
import numpy as np
import xmltodict
from PIL import Image

from fastlabel import const, converters, utils
from fastlabel.const import AnnotationType

from .api import Api
from .exceptions import FastLabelInvalidException


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

    def find_video_classification_task(self, task_id: str) -> dict:
        """
        Find a signle video classification task.
        """
        endpoint = "tasks/video/classification/" + task_id
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
        external_status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of image tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
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
        if external_status:
            params["externalStatus"] = external_status
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
        external_status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of image classification tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        endpoint = "tasks/image/classification"
        params = {"project": project}
        if status:
            params["status"] = status
        if external_status:
            params["externalStatus"] = external_status
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
        external_status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 10,
    ) -> list:
        """
        Returns a list of multi image tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
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
        if external_status:
            params["externalStatus"] = external_status
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
        external_status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 10,
    ) -> list:
        """
        Returns a list of video tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
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
        if external_status:
            params["externalStatus"] = external_status
        if tags:
            params["tags"] = tags
        if task_name:
            params["taskName"] = task_name
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_video_classification_tasks(
        self,
        project: str,
        status: str = None,
        external_status: str = None,
        tags: list = [],
        task_name: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of video classification tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        endpoint = "tasks/video/classification"
        params = {"project": project}
        if status:
            params["status"] = status
        if external_status:
            params["externalStatus"] = external_status
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
    ) -> dict:
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
        external_status: str = None,
        annotations: list = [],
        tags: list = [],
        **kwargs
    ) -> str:
        """
        Create a single image task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are png, jpg, jpeg. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        annotations is a list of annotation to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/image"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422)
        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = annotations
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_image_classification_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single image classification task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are png, jpg, jpeg. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        attributes is a list of attribute to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/image/classification"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422)
        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if attributes:
            payload["attributes"] = attributes
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_multi_image_task(
        self,
        project: str,
        name: str,
        folder_path: str,
        status: str = None,
        external_status: str = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single multi image task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        folder_path is a path to data folder. Files should be under the folder. Nested folder structure is not supported. Supported extensions of files are png, jpg, jpeg. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        annotations is a list of annotation to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        if not os.path.isdir(folder_path):
            raise FastLabelInvalidException(
                "Folder does not exist.", 422)

        endpoint = "tasks/multi-image"
        file_paths = glob.glob(os.path.join(folder_path, "*"))
        if not file_paths:
            raise FastLabelInvalidException(
                "Folder does not have any file.", 422)
        contents = []
        contents_size = 0
        for file_path in file_paths:
            if not utils.is_image_supported_ext(file_path):
                raise FastLabelInvalidException(
                    "Supported extensions are png, jpg, jpeg.", 422)

            if len(contents) == 250:
                raise FastLabelInvalidException(
                    "The count of files should be under 250", 422)

            file = utils.base64_encode(file_path)
            contents.append({
                "name": os.path.basename(file_path),
                "file": file
            })
            contents_size += utils.get_json_length(contents[-1])
            if contents_size > const.SUPPORTED_CONTENTS_SIZE:
                raise FastLabelInvalidException(
                    f"Supported contents size is under {const.SUPPORTED_CONTENTS_SIZE}.", 422)

        payload = {"project": project, "name": name, "contents": contents}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if annotations:
            payload["annotations"] = annotations
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_video_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single video task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are mp4. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        annotations is a list of annotation to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/video"
        if not utils.is_video_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are mp4.", 422)
        if os.path.getsize(file_path) > const.SUPPORTED_VIDEO_SIZE:
            raise FastLabelInvalidException(
                f"Supported video size is under 250 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = annotations
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_video_classification_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single video classification task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are mp4. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        attributes is a list of attribute to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/video/classification"
        if not utils.is_video_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are mp4.", 422)
        if os.path.getsize(file_path) > const.SUPPORTED_VIDEO_SIZE:
            raise FastLabelInvalidException(
                f"Supported video size is under 250 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if attributes:
            payload["attributes"] = attributes
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    # Task Update

    def update_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single task.

        task_id is an id of the task. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_image_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a single image task.

        task_id is an id of the task. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set. (Optional)
        annotations is a list of annotation to be set. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/image/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                # Since the content name is not passed in the sdk update api, the content will be filled on the server side.
                annotation["content"] = ""
            payload["annotations"] = annotations

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_image_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single image classification task.

        task_id is an id of the task. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        attributes is a list of attribute to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/image/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if attributes:
            payload["attributes"] = attributes
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_video_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single video classification task.

        task_id is an id of the task. (Required)
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        attributes is a list of attribute to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        assignee is slug of assigned user. (Optional)
        reviewer is slug of review user. (Optional)
        approver is slug of approve user. (Optional)
        external_assignee is slug of external assigned user. (Optional)
        external_reviewer is slug of external review user. (Optional)
        external_approver is slug of external approve user. (Optional)
        """
        endpoint = "tasks/video/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if attributes:
            payload["attributes"] = attributes
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    # Task Delete

    def delete_task(self, task_id: str) -> None:
        """
        Delete a single task.
        """
        endpoint = "tasks/" + task_id
        self.api.delete_request(endpoint)

    # Convert to Fastlabel

    def convert_coco_to_fastlabel(self, file_path: str) -> dict:
        """
        Convert COCO format to FastLabel format as annotation file.

        file_path is a COCO format annotation file. (Required)

        In the output file, the key is the image file name and the value is a list of annotations in FastLabel format, which is returned in dict format.

        output format example.
        {
            'sample1.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ],
            'sample2.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ]
        }
        """
        with open(file_path, "r") as f:
            file = f.read()
            return converters.execute_coco_to_fastlabel(eval(file))

    def convert_labelme_to_fastlabel(self, folder_path: str) -> dict:
        """
        Convert labelme format to FastLabel format as annotation files.

        folder_path is the folder that contains the labelme format files with the json extension. (Required)

        In the output file, the key is the image file name and the value is a list of annotations in FastLabel format, which is returned in dict format.
        If the tree has multiple hierarchies, the key is the relative path rooted at the specified folder name.

        output format example.
        In the case of labelme, the key is the tree structure if the tree structure is multi-level.

        [tree structure]
        dataset
        ├── sample1.jpg
        ├── sample1.json
        └── sample_dir
            ├── sample2.jpg
            └── sample2.json

        [output]
        {
            'sample1.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ],
            'sample_dir/sample2.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                    }
            ]
        }
        """
        results = {}
        for file_path in glob.iglob(
            os.path.join(folder_path, "**/**.json"), recursive=True
        ):
            with open(file_path, "r") as f:
                c = converters.execute_labelme_to_fastlabel(
                    json.load(f),
                    file_path.replace(os.path.join(*[folder_path, ""]), ""),
                )
                results[c[0]] = c[1]
        return results

    def convert_pascalvoc_to_fastlabel(self, folder_path: str) -> dict:
        """
        Convert PascalVOC format to FastLabel format as annotation files.

        folder_path is the folder that contains the PascalVOC format files with the xml extension. (Required)

        In the output file, the key is the image file name and the value is a list of annotations in FastLabel format, which is returned in dict format.
        If the tree has multiple hierarchies, the key is the relative path rooted at the specified folder name.

        output format example.
        In the case of PascalVOC, the key is the tree structure if the tree structure is multi-level.

        [tree structure]
        dataset
        ├── sample1.jpg
        ├── sample1.xml
        └── sample_dir
            ├── sample2.jpg
            └── sample2.xml

        [output]
        {
            'sample1.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ],
            'sample_dir/sample2.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ]
        }
        """
        results = {}
        for file_path in glob.iglob(
            os.path.join(folder_path, "**/**.xml"), recursive=True
        ):
            with open(file_path, "r") as f:
                file = f.read()
                c = converters.execute_pascalvoc_to_fastlabel(
                    xmltodict.parse(file),
                    file_path.replace(os.path.join(*[folder_path, ""]), ""),
                )
                results[c[0]] = c[1]
        return results

    def convert_yolo_to_fastlabel(
        self, classes_file_path: str, dataset_folder_path: str
    ) -> dict:
        """
        Convert YOLO format to FastLabel format as annotation files.

        classes_file_path is YOLO format class file. (Required)
        dataset_folder_path is the folder that contains the image file and YOLO format files with the txt extension. (Required)

        In the output file, the key is the image file name and the value is a list of annotations in FastLabel format, which is returned in dict format.
        If the tree has multiple hierarchies, the key is the relative path rooted at the specified folder name.

        output format example.
        In the case of YOLO, the key is the tree structure if the tree structure is multi-level.

        [tree structure]
        dataset
        ├── sample1.jpg
        ├── sample1.txt
        └── sample_dir
            ├── sample2.jpg
            └── sample2.txt

        [output]
        {
            'sample1.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ],
            'sample_dir/sample2.jpg':  [
                {
                    'points': [
                        100,
                        100,
                        200,
                        200
                    ],
                    'type': 'bbox',
                    'value': 'cat'
                }
            ]
        }
        """
        classes = self.__get_yolo_format_classes(classes_file_path)
        image_sizes = self.__get_yolo_image_sizes(dataset_folder_path)
        yolo_annotations = self.__get_yolo_format_annotations(
            dataset_folder_path)

        return converters.execute_yolo_to_fastlabel(
            classes,
            image_sizes,
            yolo_annotations,
            os.path.join(*[dataset_folder_path, ""]),
        )

    def __get_yolo_format_classes(self, classes_file_path: str) -> dict:
        """
        return data format
        {
            id: classs_name
            ...
        }
        """
        classes = {}
        with open(classes_file_path, "r") as f:
            lines = f.readlines()
            line_index = 0
            for line in lines:
                classes[str(line_index)] = line.strip()
                line_index += 1
        return classes

    def __get_yolo_image_sizes(self, dataset_folder_path: str) -> dict:
        """
        return data format
        {
            image_file_path_without_ext: {
                "image_file_path": image file full path
                "size": [whdth, height]
            ...
        }
        """
        image_types = utils.get_supported_image_ext()
        image_paths = [
            p for p in glob.glob(os.path.join(dataset_folder_path, "**/*"), recursive=True)
            if re.search("/*\.({})".format("|".join(image_types)), str(p))
        ]
        image_sizes = {}
        for image_path in image_paths:
            image = Image.open(image_path)
            width, height = image.size
            image_sizes[image_path.replace(os.path.splitext(image_path)[1], "")] = {
                "image_file_path": image_path,
                "size": [width, height],
            }

        return image_sizes

    def __get_yolo_format_annotations(self, dataset_folder_path: str) -> dict:
        """
        return data format
        {
            annotaion_file_path_without_ext:
                [
                    yolo_class_id,
                    yolo_center_x_ratio,
                    yolo_center_y_ratio,
                    yolo_anno_width_ratio,
                    yolo_anno_height_ratio
                ],
            ...
        }
        """
        yolo_annotations = {}
        annotaion_file_paths = [
            p for p in glob.glob(os.path.join(dataset_folder_path, "**/*.txt"), recursive=True)
            if re.search(("/*\.txt"), str(p))
        ]
        for annotaion_file_path in annotaion_file_paths:
            with open(annotaion_file_path, "r") as f:
                anno_lines = f.readlines()
                annotaion_key = annotaion_file_path.replace(".txt", "")
                yolo_annotations[annotaion_key] = []
                for anno_line in anno_lines:
                    yolo_annotations[annotaion_key].append(
                        anno_line.strip().split(" "))
        return yolo_annotations

    # Task Convert

    def export_coco(self, tasks: list, annotations: list = [], output_dir: str = os.path.join("output", "coco"), output_file_name: str = "annotations.json") -> None:
        """
        Convert tasks to COCO format and export as a file.
        If you pass annotations, you can export Pose Estimation type annotations.

        tasks is a list of tasks. (Required)
        annotations is a list of annotations. (Optional)
        output_dir is output directory(default: output/coco). (Optional)
        output_file_name is output file name(default: annotations.json). (Optional)
        """
        if not utils.is_json_ext(output_file_name):
            raise FastLabelInvalidException(
                "Output file name must have a json extension", 422)
        coco = converters.to_coco(tasks, annotations)
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, output_file_name)
        with open(file_path, 'w') as f:
            json.dump(coco, f, indent=4, ensure_ascii=False)

    def export_yolo(self, tasks: list, classes: list = [], output_dir: str = os.path.join("output", "yolo")) -> None:
        """
        Convert tasks to YOLO format and export as files.
        If you pass classes, classes.txt will be generated based on it .
        If not , classes.txt will be generated based on passed tasks .(Annotations never used in your project will not be exported.)

        tasks is a list of tasks. (Required)
        classes is a list of annotation values.  e.g. ['dog','bird'] (Optional)
        output_dir is output directory(default: output/yolo). (Optional)
        """
        annos, categories = converters.to_yolo(tasks, classes)
        for anno in annos:
            file_name = anno["filename"]
            basename = utils.get_basename(file_name)
            file_path = os.path.join(
                output_dir, "annotations", basename + ".txt")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding="utf8") as f:
                for obj in anno["object"]:
                    f.write(obj)
                    f.write("\n")
        classes_file_path = os.path.join(output_dir, "classes.txt")
        os.makedirs(os.path.dirname(classes_file_path), exist_ok=True)
        with open(classes_file_path, 'w', encoding="utf8") as f:
            for category in categories:
                f.write(category["name"])
                f.write("\n")

    def export_pascalvoc(self, tasks: list, output_dir: str = os.path.join("output", "pascalvoc")) -> None:
        """
        Convert tasks to Pascal VOC format as files.

        tasks is a list of tasks. (Required)
        output_dir is output directory(default: output/pascalvoc). (Optional)
        """
        pascalvoc = converters.to_pascalvoc(tasks)
        for voc in pascalvoc:
            file_name = voc["annotation"]["filename"]
            basename = utils.get_basename(file_name)
            file_path = os.path.join(output_dir, basename + ".xml")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            xml = xmltodict.unparse(voc, pretty=True, full_document=False)
            with open(file_path, 'w', encoding="utf8") as f:
                f.write(xml)

    def export_labelme(self, tasks: list, output_dir: str = os.path.join("output", "labelme")) -> None:
        """
        Convert tasks to labelme format as files.

        tasks is a list of tasks. (Required)
        output_dir is output directory(default: output/labelme). (Optional)
        """
        labelmes = converters.to_labelme(tasks)
        for labelme in labelmes:
            file_name = labelme["imagePath"]
            basename = utils.get_basename(file_name)
            file_path = os.path.join(output_dir, basename + ".json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(labelme, f, indent=4, ensure_ascii=False)

    # Instance / Semantic Segmetation

    def export_instance_segmentation(self, tasks: list, output_dir: str = os.path.join("output", "instance_segmentation"), pallete: List[int] = const.COLOR_PALETTE) -> None:
        """
        Convert tasks to index color instance segmentation (PNG files).
        Supports only bbox, polygon and segmentation annotation types.
        Supports up to 57 instances in default colors palette. Check const.COLOR_PALETTE for more details.

        tasks is a list of tasks. (Required)
        output_dir is output directory(default: output/instance_segmentation). (Optional)
        pallete is color palette of index color. Ex: [255, 0, 0, ...] (Optional)
        """
        tasks = converters.to_pixel_coordinates(tasks)
        for task in tasks:
            self.__export_index_color_image(
                task=task, output_dir=output_dir, pallete=pallete, is_instance_segmentation=True)

    def export_semantic_segmentation(self, tasks: list, output_dir: str = os.path.join("output", "semantic_segmentation"), pallete: List[int] = const.COLOR_PALETTE) -> None:
        """
        Convert tasks to index color semantic segmentation (PNG files).
        Supports only bbox, polygon and segmentation annotation types.
        Check const.COLOR_PALETTE for color pallete.

        tasks is a list of tasks. (Required)
        output_dir is output directory(default: output/semantic_segmentation). (Optional)
        pallete is color palette of index color. Ex: [255, 0, 0, ...] (Optional)
        """
        classes = []
        for task in tasks:
            for annotation in task["annotations"]:
                classes.append(annotation["value"])
        classes = list(set(classes))
        classes.sort()

        tasks = converters.to_pixel_coordinates(tasks)
        for task in tasks:
            self.__export_index_color_image(
                task=task, output_dir=output_dir, pallete=pallete, is_instance_segmentation=False, classes=classes)

    def __export_index_color_image(self, task: list, output_dir: str, pallete: List[int], is_instance_segmentation: bool = True, classes: list = []) -> None:
        image = Image.new("RGB", (task["width"], task["height"]), 0)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        index = 1
        # In case segmentation, to avoid hollowed points overwrite other segmentation in them, segmentation rendering process is different from other annotation type
        seg_mask_images = []
        for annotation in task["annotations"]:
            color = index if is_instance_segmentation else classes.index(
                annotation["value"]) + 1
            if annotation["type"] == AnnotationType.segmentation.value:
                # Create each annotation's masks and merge them finally
                seg_mask_ground = Image.new(
                    "RGB", (task["width"], task["height"]), 0)
                seg_mask_image = np.array(seg_mask_ground)
                seg_mask_image = cv2.cvtColor(seg_mask_image, cv2.COLOR_BGR2GRAY)

                for region in annotation["points"]:
                    count = 0
                    for points in region:
                        if count == 0:
                            cv_draw_points = []
                            if utils.is_clockwise(points):
                                cv_draw_points = self.__get_cv_draw_points(
                                    points)
                            else:
                                cv_draw_points = self.__get_cv_draw_points(
                                    utils.reverse_points(points))
                            cv2.fillPoly(
                                seg_mask_image, [cv_draw_points], color, lineType=cv2.LINE_8, shift=0)
                        else:
                            # Reverse hollow points for opencv because this points are counter clockwise
                            cv_draw_points = self.__get_cv_draw_points(
                                utils.reverse_points(points))
                            cv2.fillPoly(
                                seg_mask_image, [cv_draw_points], 0, lineType=cv2.LINE_8, shift=0)
                        count += 1
                seg_mask_images.append(seg_mask_image)
            elif annotation["type"] == AnnotationType.polygon.value:
                cv_draw_points = self.__get_cv_draw_points(
                    annotation["points"])
                cv2.fillPoly(image, [cv_draw_points], color,
                             lineType=cv2.LINE_8, shift=0)
            elif annotation["type"] == AnnotationType.bbox.value:
                cv_draw_points = self.__get_cv_draw_points(
                    annotation["points"])
                cv2.fillPoly(image, [cv_draw_points], color,
                             lineType=cv2.LINE_8, shift=0)
            else:
                continue
            index += 1

        # For segmentation, merge each mask images
        for seg_mask_image in seg_mask_images:
            image = image | seg_mask_image

        image_path = os.path.join(
            output_dir, utils.get_basename(task["name"]) + ".png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image)
        image = image.convert('P')
        image.putpalette(pallete)
        image.save(image_path)

    def __get_cv_draw_points(self, points: List[int]) -> List[int]:
        """
        Convert points to pillow draw points. Diagonal points are not supported.
        """
        x_points = []
        x_points.append(points[0])
        x_points.append(points[1])
        for i in range(int(len(points) / 2)):
            if i == 0:
                continue
            x = points[i * 2]
            y = points[i * 2 + 1]
            if y > x_points[(i - 1) * 2 + 1]:
                x_points[(i - 1) * 2] = x_points[(i - 1) * 2] - 1
                x = x - 1
            x_points.append(x)
            x_points.append(y)

        y_points = []
        y_points.append(points[0])
        y_points.append(points[1])
        for i in range(int(len(points) / 2)):
            if i == 0:
                continue
            x = points[i * 2]
            y = points[i * 2 + 1]
            if x < y_points[(i - 1) * 2]:
                y_points[(i - 1) * 2 + 1] = y_points[(i - 1) * 2 + 1] - 1
                y = y - 1
            y_points.append(x)
            y_points.append(y)

        new_points = []
        for i in range(int(len(points) / 2)):
            new_points.append(x_points[i * 2])
            new_points.append(y_points[i * 2 + 1])

        cv_points = []
        for i in range(int(len(new_points) / 2)):
            cv_points.append((new_points[i * 2], new_points[i * 2 + 1]))
        return np.array(cv_points)

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
        color: str = None,
        order: int = None,
        attributes: list = []
    ) -> str:
        """
        Create an annotation.

        project is slug of your project. (Required)
        type can be 'bbox', 'polygon', 'keypoint', 'classification', 'line', 'segmentation'. (Required)
        value is an unique identifier of annotation in your project. (Required)
        title is a display name of value. (Required)
        color is hex color code like #ffffff. (Optional)
        attributes is a list of attribute. (Optional)
        """
        endpoint = "annotations"
        payload = {
            "project": project,
            "type": type,
            "value": value,
            "title": title,
        }
        if color:
            payload["color"] = color
        if order:
            payload["order"] = order
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
        order: int = None,
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
        if order:
            payload["order"] = order
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

    # Project

    def find_project(self, project_id: str) -> dict:
        """
        Find a project.
        """
        endpoint = "projects/" + project_id
        return self.api.get_request(endpoint)

    def find_project_by_slug(self, slug: str) -> dict:
        """
        Find a project by slug.

        slug is slug of your project. (Required)
        """
        projects = self.get_projects(slug=slug)
        if not projects:
            return None
        return projects[0]

    def get_projects(
        self,
        slug: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of projects.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        slug is slug of your project. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422)
        endpoint = "projects"
        params = {}
        if slug:
            params["slug"] = slug
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_project_id_slug_map(
        self,
        offset: int = None,
        limit: int = 1000,
    ) -> dict:
        """
        Returns a map of project ids and slugs.
        e.g.) {
                "88e74507-07b5-4607-a130-cb6316ca872c", "image-bbox-slug",
                "fe2c24a4-8270-46eb-9c78-bb7281c8bdgs", "image-video-slug"
              }
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422)
        endpoint = "projects/map/id-slug"
        params = {}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def create_project(
        self,
        type: str,
        name: str,
        slug: str,
        is_pixel: bool = False,
        job_size: int = 10,
        workflow: str = "two_step",
        external_workflow: str = "two_step",
    ) -> str:
        """
        Create a project.

        type can be 'image_bbox', 'image_polygon', 'image_keypoint', 'image_line', 'image_segmentation', 'image_classification', 'image_all', 'multi_image_bbox', 'multi_image_polygon', 'multi_image_keypoint', 'multi_image_line', 'multi_image_segmentation', 'video_bbox', 'video_single_classification'. (Required)
        name is name of your project. (Required)
        slug is slug of your project. (Required)
        is_pixel is whether to annotate image with pixel level. (Optional)
        job_size is the number of tasks the annotator gets at one time. (Optional)
        workflow is the type of annotation wokflow. workflow can be 'two_step' or 'three_step' (Optional)
        external_workflow is the type of external annotation wokflow. external_workflow can be 'two_step' or 'three_step' (Optional)
        """
        endpoint = "projects"
        payload = {
            "type": type,
            "name": name,
            "slug": slug,
            "workflow": workflow,
            "externalWorkflow": external_workflow,
        }
        if is_pixel:
            payload["isPixel"] = is_pixel
        if job_size:
            payload["jobSize"] = job_size
        return self.api.post_request(endpoint, payload=payload)

    def update_project(
        self,
        project_id: str,
        name: str = None,
        slug: str = None,
        job_size: int = None,
        workflow: str = None,
        external_workflow: str = None,
    ) -> str:
        """
        Update a project.

        project_id is an id of the project. (Required)
        name is name of your project. (Optional)
        slug is slug of your project. (Optional)
        job_size is the number of tasks the annotator gets at one time. (Optional)
        workflow is the type of annotation wokflow. workflow can be 'two_step' or 'three_step' (Optional)
        external_workflow is the type of external annotation wokflow. external_workflow can be 'two_step' or 'three_step' (Optional)
        """
        endpoint = "projects/" + project_id
        payload = {}
        if name:
            payload["name"] = name
        if slug:
            payload["slug"] = slug
        if job_size:
            payload["jobSize"] = job_size
        if workflow:
            payload["workflow"] = workflow
        if external_workflow:
            payload["externalWorkflow"] = external_workflow
        return self.api.put_request(endpoint, payload=payload)

    def delete_project(self, project_id: str) -> None:
        """
        Delete a project.
        """
        endpoint = "projects/" + project_id
        self.api.delete_request(endpoint)

    @staticmethod
    def __fill_assign_users(payload: dict, **kwargs):
        if "assignee" in kwargs:
            payload["assignee"] = kwargs.get("assignee")
        if "reviewer" in kwargs:
            payload["reviewer"] = kwargs.get("reviewer")
        if "approver" in kwargs:
            payload["approver"] = kwargs.get("approver")
        if "external_assignee" in kwargs:
            payload["externalAssignee"] = kwargs.get("external_assignee")
        if "external_reviewer" in kwargs:
            payload["externalReviewer"] = kwargs.get("external_reviewer")
        if "external_approver" in kwargs:
            payload["externalApprover"] = kwargs.get("external_approver")
