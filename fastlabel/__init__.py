import glob
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import cv2
import numpy as np
import requests
import xmltodict
from PIL import Image, ImageColor, ImageDraw

from fastlabel import const, converters, utils
from fastlabel.const import (
    EXPORT_IMAGE_WITH_ANNOTATIONS_SUPPORTED_IMAGE_TYPES,
    KEYPOINT_MIN_STROKE_WIDTH,
    OPACITY_DARK,
    OPACITY_THIN,
    POSE_ESTIMATION_MIN_STROKE_WIDTH,
    SEPARATOER,
    AnnotationType,
    DatasetObjectType,
    Priority,
)

from .api import Api
from .exceptions import FastLabelInvalidException
from .query import DatasetObjectGetQuery

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


class Client:
    api = None

    def __init__(self):
        self.api = Api()

    # Task Find

    def find_image_task(self, task_id: str) -> dict:
        """
        Find a single image task.
        """
        endpoint = "tasks/image/" + task_id
        return self.api.get_request(endpoint)

    def find_image_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single image task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_image_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_image_classification_task(self, task_id: str) -> dict:
        """
        Find a single image classification task.
        """
        endpoint = "tasks/image/classification/" + task_id
        return self.api.get_request(endpoint)

    def find_multi_image_classification_task(self, task_id: str) -> dict:
        """
        Find a single multi image classification task.
        """
        endpoint = "tasks/multi-image/classification/" + task_id
        return self.api.get_request(endpoint)

    def find_multi_image_classification_task_by_name(
        self, project: str, task_name: str
    ) -> dict:
        """
        Find a single multi image classification task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_multi_image_classification_tasks(
            project=project, task_name=task_name
        )
        if not tasks:
            return None
        return tasks[0]

    def find_image_classification_task_by_name(
        self, project: str, task_name: str
    ) -> dict:
        """
        Find a single image classification task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_image_classification_tasks(
            project=project, task_name=task_name
        )
        if not tasks:
            return None
        return tasks[0]

    def find_sequential_image_task(self, task_id: str) -> dict:
        """
        Find a single sequential image task.
        """
        endpoint = "tasks/sequential-image/" + task_id
        return self.api.get_request(endpoint)

    def find_sequential_image_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single sequential image task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_sequential_image_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_video_task(self, task_id: str) -> dict:
        """
        Find a single video task.
        """
        endpoint = "tasks/video/" + task_id
        return self.api.get_request(endpoint)

    def find_video_classification_task(self, task_id: str) -> dict:
        """
        Find a single video classification task.
        """
        endpoint = "tasks/video/classification/" + task_id
        return self.api.get_request(endpoint)

    def find_video_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single video task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_video_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_video_classification_task_by_name(
        self, project: str, task_name: str
    ) -> dict:
        """
        Find a single video classification task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_video_classification_tasks(
            project=project, task_name=task_name
        )
        if not tasks:
            return None
        return tasks[0]

    def find_text_task(self, task_id: str) -> dict:
        """
        Find a single text task.
        """
        endpoint = "tasks/text/" + task_id
        return self.api.get_request(endpoint)

    def find_text_classification_task(self, task_id: str) -> dict:
        """
        Find a single text classification task.
        """
        endpoint = "tasks/text/classification/" + task_id
        return self.api.get_request(endpoint)

    def find_text_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single text task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_text_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_text_classification_task_by_name(
        self, project: str, task_name: str
    ) -> dict:
        """
        Find a single text classification task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_text_classification_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_audio_task(self, task_id: str) -> dict:
        """
        Find a single audio task.
        """
        endpoint = "tasks/audio/" + task_id
        return self.api.get_request(endpoint)

    def find_audio_classification_task(self, task_id: str) -> dict:
        """
        Find a single audio classification task.
        """
        endpoint = "tasks/audio/classification/" + task_id
        return self.api.get_request(endpoint)

    def find_audio_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single audio task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_audio_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_audio_classification_task_by_name(
        self, project: str, task_name: str
    ) -> dict:
        """
        Find a single audio classification task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_audio_classification_tasks(
            project=project, task_name=task_name
        )
        if not tasks:
            return None
        return tasks[0]

    def find_dicom_task(self, task_id: str) -> dict:
        """
        Find a single DICOM task.
        """
        endpoint = "tasks/dicom/" + task_id
        return self.api.get_request(endpoint)

    def find_dicom_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single DICOM task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_dicom_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_pcd_task(self, task_id: str) -> dict:
        """
        Find a single PCD task.
        """
        endpoint = "tasks/pcd/" + task_id
        return self.api.get_request(endpoint)

    def find_pcd_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single PCD task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_pcd_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_sequential_pcd_task(self, task_id: str) -> dict:
        """
        Find a single Sequential PCD task.
        """
        endpoint = "tasks/sequential-pcd/" + task_id
        return self.api.get_request(endpoint)

    def find_sequential_pcd_task_by_name(self, project: str, task_name: str) -> dict:
        """
        Find a single Sequential PCD task by name.

        project is slug of your project (Required).
        task_name is a task name (Required).
        """
        tasks = self.get_sequential_pcd_tasks(project=project, task_name=task_name)
        if not tasks:
            return None
        return tasks[0]

    def find_history(self, history_id: str) -> dict:
        """
        Find a single history.
        """
        endpoint = "tasks/import/histories/" + history_id
        return self.api.get_request(endpoint)

    # Task Get

    def count_tasks(
        self,
        project: str,
        status: str = None,
        external_status: str = None,
        tags: list = None,
    ) -> int:
        """
        Returns task count.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined',
        'customer_declined' (Optional).
        tags is a list of tag (Optional).
        """
        endpoint = "tasks/count"
        params = {"project": project}
        if status:
            params["status"] = status
        if external_status:
            params["externalStatus"] = external_status
        if tags:
            params["tags"] = tags
        return self.api.get_request(endpoint, params=params)

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
        Returns up to 1000 at a time, to get more,
        set offset as the starting position to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined',
        'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
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
        Returns up to 1000 at a time, to get more,
        set offset as the starting position to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
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

    def get_multi_image_classification_tasks(
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
        Returns a list of multi image classification tasks.
        Returns up to 1000 at a time, to get more,
        set offset as the starting position to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        endpoint = "tasks/multi-image/classification"
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

    def get_sequential_image_tasks(
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
        Returns a list of sequential image tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 10:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 10.", 422
            )
        endpoint = "tasks/sequential-image"
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
        Returns up to 10 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 10:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 10.", 422
            )
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
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
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

    def get_text_tasks(
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
        Returns a list of text tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tasks/text"
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

    def get_text_classification_tasks(
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
        Returns a list of text classification tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        endpoint = "tasks/text/classification"
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

    def get_audio_tasks(
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
        Returns a list of audio tasks.
        Returns up to 10 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tasks/audio"
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

    def get_audio_classification_tasks(
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
        Returns a list of audio classification tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        endpoint = "tasks/audio/classification"
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

    def get_pcd_tasks(
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
        Returns a list of PCD tasks.
        Returns up to 1000 at a time, to get more,
        set offset as the starting position to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined',
        'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tasks/pcd"
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

    def get_sequential_pcd_tasks(
        self,
        project: str,
        status: str = None,
        external_status: str = None,
        tags: list = None,
        task_name: str = None,
        offset: int = None,
        limit: int = 10,
    ) -> list:
        """
        Returns a list of Sequential PCD tasks.
        Returns up to 10 at a time, to get more,
        set offset as the starting position to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined',
        'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 10:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 10.", 422
            )
        endpoint = "tasks/sequential-pcd"
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
        self,
        project: str,
        offset: int = None,
        limit: int = 1000,
    ) -> dict:
        """
        Returns a map of task ids and names.
        e.g.) {
                "88e74507-07b5-4607-a130-cb6316ca872c", "01_cat.jpg",
                "fe2c24a4-8270-46eb-9c78-bb7281c8bdgs", "02_cat.jpg"
              }
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tasks/map/id-name"
        params = {"project": project}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def get_dicom_tasks(
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
        Returns a list of DICOM tasks.
        Returns up to 1000 at a time, to get more,
        set offset as the starting position to fetch.

        project is slug of your project (Required).
        status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined'. (Optional)
        external_status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined',
        'customer_declined' (Optional).
        tags is a list of tag (Optional).
        task_name is a task name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tasks/dicom"
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

    # Task Create

    def create_image_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        custom_task_status: str = "",
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        is_delete_exif: bool = False,
        **kwargs,
    ) -> str:
        """
        Create a single image task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are png, jpg, jpeg (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        custom_task_status is the initial value of the status.
        it can be set to 'registered', 'pending', 'workflow_{1..6}_completed', 'workflow_{1..6}_declined' (Optional).
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        workflow_{1..6}_user is slug of custom workflow assignee user for each step (Optional).
        """
        endpoint = "tasks/image"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422
            )
        if not utils.is_image_supported_size(file_path):
            raise FastLabelInvalidException("Supported image size is under 20 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if tags:
            payload["tags"] = tags
        if is_delete_exif:
            payload["isDeleteExif"] = is_delete_exif
        if custom_task_status:
            payload["customTaskStatus"] = custom_task_status

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_integrated_image_task(
        self,
        project: str,
        storage_type: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        annotations: list = None,
        tags: list = None,
        **kwargs,
    ) -> str:
        """
        Create a single integrated image task.

        project is slug of your project (Required).
        storage type is the type of storage where your file resides (Required). e.g.) gcp
        file_path is a path to data in your setting storage bucket. Supported extensions are png, jpg, jpeg (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/integrated-image"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422
            )

        payload = {
            "project": project,
            "filePath": file_path,
            "storageType": storage_type,
        }
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if annotations:
            for annotation in annotations:
                annotation["content"] = file_path
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
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
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        is_delete_exif: bool = False,
        **kwargs,
    ) -> str:
        """
        Create a single image classification task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are png, jpg, jpeg (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/image/classification"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422
            )
        if not utils.is_image_supported_size(file_path):
            raise FastLabelInvalidException("Supported image size is under 20 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        if is_delete_exif:
            payload["isDeleteExif"] = is_delete_exif

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_integrated_image_classification_task(
        self,
        project: str,
        file_path: str,
        storage_type: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = None,
        tags: list = None,
        **kwargs,
    ) -> str:
        """
        Create a single integrated image classification task.

        project is slug of your project (Required).
        storage type is the type of storage where your file resides (Required). e.g.) gcp
        file_path is a path to data in your setting storage bucket. Supported extensions are png, jpg, jpeg (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/integrated-image/classification"
        payload = {
            "project": project,
            "filePath": file_path,
            "storageType": storage_type,
        }
        attributes = attributes or []
        tags = tags or []
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_multi_image_classification_task(
        self,
        project: str,
        name: str,
        folder_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        is_delete_exif: bool = False,
        **kwargs,
    ) -> str:
        """
        Create a single multi image classification task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        folder_path is a path to data folder. Files should be under the folder.
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/multi-image/classification"
        if not os.path.isdir(folder_path):
            raise FastLabelInvalidException("Folder does not exist.", 422)
        file_paths = glob.glob(os.path.join(folder_path, "*"))
        if not file_paths:
            raise FastLabelInvalidException("Folder does not have any file.", 422)
        contents = []
        contents_size = 0
        for file_path in file_paths:
            if not utils.is_image_supported_ext(file_path):
                raise FastLabelInvalidException(
                    "Supported extensions are png, jpg, jpeg.", 422
                )

            if not utils.is_image_supported_size(file_path):
                raise FastLabelInvalidException(
                    "Supported image size is under 20 MB.", 422
                )

            if len(contents) == 6:
                raise FastLabelInvalidException(
                    "The count of files should be under 6", 422
                )

            file = utils.base64_encode(file_path)
            contents.append({"name": os.path.basename(file_path), "file": file})
            contents_size += utils.get_json_length(contents[-1])
            if contents_size > const.SUPPORTED_CONTENTS_SIZE:
                raise FastLabelInvalidException(
                    "Supported contents size is under"
                    f" {const.SUPPORTED_CONTENTS_SIZE}.",
                    422,
                )

        payload = {"project": project, "name": name, "contents": contents}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags
        if is_delete_exif:
            payload["isDeleteExif"] = is_delete_exif

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_sequential_image_task(
        self,
        project: str,
        name: str,
        folder_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        is_delete_exif: bool = False,
        **kwargs,
    ) -> str:
        """
        Create a single sequential image task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        folder_path is a path to data folder. Files should be under the folder.
        Nested folder structure is not supported. Supported extensions of files
        are png, jpg, jpeg (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        if not os.path.isdir(folder_path):
            raise FastLabelInvalidException("Folder does not exist.", 422)

        endpoint = "tasks/sequential-image"
        file_paths = glob.glob(os.path.join(folder_path, "*"))
        if not file_paths:
            raise FastLabelInvalidException("Folder does not have any file.", 422)
        contents = []
        contents_size = 0
        for file_path in file_paths:
            if not utils.is_image_supported_ext(file_path):
                raise FastLabelInvalidException(
                    "Supported extensions are png, jpg, jpeg.", 422
                )

            if not utils.is_image_supported_size(file_path):
                raise FastLabelInvalidException(
                    "Supported image size is under 20 MB.", 422
                )

            if len(contents) == 250:
                raise FastLabelInvalidException(
                    "The count of files should be under 250", 422
                )

            file = utils.base64_encode(file_path)
            contents.append({"name": os.path.basename(file_path), "file": file})
            contents_size += utils.get_json_length(contents[-1])
            if contents_size > const.SUPPORTED_CONTENTS_SIZE:
                raise FastLabelInvalidException(
                    "Supported contents size is under"
                    f" {const.SUPPORTED_CONTENTS_SIZE}.",
                    422,
                )

        payload = {"project": project, "name": name, "contents": contents}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if tags:
            payload["tags"] = tags
        if is_delete_exif:
            payload["isDeleteExif"] = is_delete_exif

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_video_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single video task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are mp4 (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/video"
        if not utils.is_video_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are mp4.", 422)
        if not utils.is_video_supported_size(file_path):
            raise FastLabelInvalidException(
                "Supported video size is under 250 MB.", 422
            )
        if not utils.is_video_supported_codec(file_path):
            raise FastLabelInvalidException(
                "Supported video encoding for registration through the SDK is only AVC/H.264",
                422,
            )

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
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
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single video classification task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are mp4 (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/video/classification"
        if not utils.is_video_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are mp4.", 422)
        if not utils.is_video_supported_size(file_path):
            raise FastLabelInvalidException(
                "Supported video size is under 250 MB.", 422
            )
        if not utils.is_video_supported_codec(file_path):
            raise FastLabelInvalidException(
                "Supported video encoding for registration through the SDK is only AVC/H.264",
                422,
            )

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_text_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single text task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are txt (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/text"
        if not utils.is_text_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are txt.", 422)
        if not utils.is_text_supported_size(file_path):
            raise FastLabelInvalidException("Supported text size is under 2 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_text_classification_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single text classification task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are mp4 (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/text/classification"
        if not utils.is_text_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are txt.", 422)
        if not utils.is_text_supported_size(file_path):
            raise FastLabelInvalidException("Supported text size is under 2 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_audio_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single audio task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are mp4 (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/audio"
        if not utils.is_audio_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are mp3, wav and w4a.", 422
            )
        if not utils.is_audio_supported_size(file_path):
            raise FastLabelInvalidException(
                "Supported audio size is under 120 MB.", 422
            )

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_audio_classification_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single audio classification task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are mp4 (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/audio/classification"
        if not utils.is_audio_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are mp3, wav and w4a.", 422
            )
        if not utils.is_audio_supported_size(file_path):
            raise FastLabelInvalidException(
                "Supported audio size is under 120 MB.", 422
            )

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_dicom_task(
        self,
        project: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single dicom task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are png, jpg, jpeg (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/dicom"
        if not utils.is_dicom_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are zip.", 422)
        if not utils.is_dicom_supported_size(file_path):
            raise FastLabelInvalidException("Supported image size is under 2 GB.", 422)

        payload = {"project": project}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        signed_url = self.__get_signed_path(
            project=project,
            file_name=os.path.basename(file_path),
            file_type="application/zip",
        )
        self.api.upload_zipfile(url=signed_url["url"], file_path=file_path)

        payload["fileKey"] = signed_url["name"]
        return self.api.post_request(endpoint, payload=payload)

    def create_pcd_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single PCD task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        file_path is a path to data. Supported extensions are pcd only (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/pcd"
        if not utils.is_pcd_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are pcd only", 422)
        if not utils.is_pcd_supported_size(file_path):
            raise FastLabelInvalidException("Supported PCD size is under 100 MB.", 422)

        file = utils.base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def create_sequential_pcd_task(
        self,
        project: str,
        name: str,
        folder_path: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        annotations: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single sequential PCD task.

        project is slug of your project (Required).
        name is an unique identifier of task in your project (Required).
        folder_path is a path to data folder. Files should be under the folder.
        Nested folder structure is not supported. Supported extensions are
        pcd only (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        annotations is a list of annotation to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        if not os.path.isdir(folder_path):
            raise FastLabelInvalidException("Folder does not exist.", 422)

        endpoint = "tasks/sequential-pcd"
        file_paths = glob.glob(os.path.join(folder_path, "*"))
        if not file_paths:
            raise FastLabelInvalidException("Folder does not have any file.", 422)
        contents = []
        contents_size = 0
        for file_path in file_paths:
            if not utils.is_pcd_supported_ext(file_path):
                raise FastLabelInvalidException(
                    "Supported extensions are pcd only", 422
                )

            if not utils.is_pcd_supported_size(file_path):
                raise FastLabelInvalidException(
                    "Supported PCD size is under 30 MB.", 422
                )

            if len(contents) == 250:
                raise FastLabelInvalidException(
                    "The count of files should be under 250", 422
                )

            file = utils.base64_encode(file_path)
            contents.append({"name": os.path.basename(file_path), "file": file})
            contents_size += utils.get_json_length(contents[-1])
            if contents_size > const.SUPPORTED_CONTENTS_SIZE:
                raise FastLabelInvalidException(
                    "Supported contents size is under"
                    f" {const.SUPPORTED_CONTENTS_SIZE}.",
                    422,
                )

        payload = {"project": project, "name": name, "contents": contents}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if annotations:
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.post_request(endpoint, payload=payload)

    def import_appendix_file(
        self,
        project: str,
        file_path: str,
    ) -> list:
        """
        Import calibration file zip.
            project is slug of your project (Required).
            file_path is a path to data. Supported extensions are zip (Required).
        """

        if not utils.is_appendix_supported_ext(file_path):
            raise FastLabelInvalidException("Supported extensions are zip.", 422)

        endpoint = "contents/imports/appendix/batch"
        payload = {"project": project}
        signed_url = self.__get_signed_path(
            project=project,
            file_name=os.path.basename(file_path),
            file_type="application/zip",
        )
        self.api.upload_zipfile(url=signed_url["url"], file_path=file_path)
        payload["fileKey"] = signed_url["name"]

        return self.api.post_request(endpoint, payload=payload)

    # Task Update

    def update_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_image_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        custom_task_status: str = "",
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        relations: Optional[List[dict]] = None,
        **kwargs,
    ) -> str:
        """
        Update a single image task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        custom_task_status is the initial value of the status.
        it can be set to 'registered', 'pending', 'workflow_{1..6}_completed', 'workflow_{1..6}_declined' (Optional).
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        relations is a list of annotation relations to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        workflow_{1..6}_user is slug of custom workflow assignee user for each step (Optional).
        """
        endpoint = "tasks/image/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                # Since the content name is not passed in the sdk update api,
                # the content will be filled on the server side.
                annotation["content"] = ""
            payload["annotations"] = delete_extra_annotations_parameter(annotations)
        if relations:
            payload["relations"] = relations
        if custom_task_status:
            payload["customTaskStatus"] = custom_task_status

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_image_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Create a single image classification task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/image/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_multi_image_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single multi image classification task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/multi-image/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_sequential_image_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a sequential image task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/sequential-image/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            payload["annotations"] = delete_extra_annotations_parameter(annotations)

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_video_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a single video task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/video/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                annotation["content"] = ""
            payload["annotations"] = delete_extra_annotations_parameter(annotations)

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_video_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single video classification task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/video/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_text_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a single text task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/text/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                annotation["content"] = ""
            payload["annotations"] = delete_extra_annotations_parameter(annotations)

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_text_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single text classification task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/text/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_audio_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a single audio task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/audio/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                annotation["content"] = ""
            payload["annotations"] = delete_extra_annotations_parameter(annotations)

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_audio_classification_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        attributes: list = [],
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single audio classification task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined' (Optional).
        attributes is a list of attribute to be set in advance (Optional).
        tags is a list of tag to be set in advance (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/audio/classification/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if attributes:
            payload["attributes"] = delete_extra_attributes_parameter(attributes)
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_pcd_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a single pcd task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/pcd/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                # Since the content name is not passed in the sdk update api,
                # the content will be filled on the server side.
                annotation["content"] = ""
            payload["annotations"] = delete_extra_annotations_parameter(annotations)

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def update_sequential_pcd_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        priority: Priority = None,
        tags: list = [],
        annotations: List[dict] = [],
        **kwargs,
    ) -> str:
        """
        Update a single sequential PCD task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set (Optional).
        annotations is a list of annotation to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/sequential-pcd/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if priority is not None:
            payload["priority"] = priority
        if tags:
            payload["tags"] = tags
        if annotations:
            for annotation in annotations:
                # Since the content name is not passed in the sdk update api,
                # the content will be filled on the server side.
                annotation["content"] = ""
            payload["annotations"] = delete_extra_annotations_parameter(annotations)

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    # Task Delete

    def delete_task(self, task_id: str) -> None:
        """
        Delete a single task.
        """
        endpoint = "tasks/" + task_id
        self.api.delete_request(endpoint)

    # Task Annotations Delete

    def delete_task_annotations(self, task_id: str) -> None:
        """
        Delete annotations in a task.
        """
        endpoint = "tasks/" + task_id + "/task-annotations"
        self.api.delete_request(endpoint)

    # Integrate Task

    def find_integrated_image_task_by_prefix(
        self,
        project: str,
        prefix: str,
    ) -> dict:
        """
        Returns a integrate image task.
        project is slug of your project (Required).
        prefix is a prefix of task name (Required).
        """
        endpoint = "tasks/integrate/images"
        params = {"project": project, "prefix": prefix}

        return self.api.get_request(endpoint, params=params)

    def find_integrated_video_task_by_prefix(
        self,
        project: str,
        prefix: str,
    ) -> dict:
        """
        Returns a integrate video task.

        project is slug of your project (Required).
        prefix is a prefix of task name (Required).
        """
        endpoint = "tasks/integrate/videos"
        params = {"project": project, "prefix": prefix}

        return self.api.get_request(endpoint, params=params)

    def find_integrated_audio_task_by_prefix(
        self,
        project: str,
        prefix: str,
    ) -> dict:
        """
        Returns a integrate audio task.

        project is slug of your project (Required).
        prefix is a prefix of task name (Required).
        """
        endpoint = "tasks/integrate/audios"
        params = {"project": project, "prefix": prefix}

        return self.api.get_request(endpoint, params=params)

    # Convert to Fastlabel

    def convert_coco_to_fastlabel(self, file_path: str, annotation_type: str) -> dict:
        """
        Convert COCO format to FastLabel format as annotation file.

        file_path is a COCO format annotation file (Required).

        In the output file, the key is the image file name and the value is a list of
        annotations in FastLabel format, which is returned in dict format.

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
        return converters.execute_coco_to_fastlabel(
            json.load(open(file_path)), annotation_type
        )

    def convert_labelme_to_fastlabel(self, folder_path: str) -> dict:
        """
        Convert labelme format to FastLabel format as annotation files.

        folder_path is the folder that contains the labelme format files
        with the json extension (Required).

        In the output file, the key is the image file name and the value is a
        list of annotations in FastLabel format, which is returned in dict format.
        If the tree has multiple hierarchies, the key is the relative path rooted
        at the specified folder name.

        output format example.
        In the case of labelme, the key is the tree structure
        if the tree structure is multi-level.

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

        folder_path is the folder that contains the PascalVOC format files with
        the xml extension (Required).

        In the output file, the key is the image file name and the value is
        a list of annotations in FastLabel format, which is returned in dict format.
        If the tree has multiple hierarchies, the key is the relative path
        rooted at the specified folder name.

        output format example.
        In the case of PascalVOC, the key is the tree structure
        if the tree structure is multi-level.

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

        classes_file_path is YOLO format class file (Required).
        dataset_folder_path is the folder that contains the image file and
        YOLO format files with the txt extension (Required).

        In the output file, the key is the image file name and the value is a
        list of annotations in FastLabel format, which is returned in dict format.
        If the tree has multiple hierarchies, the key is the relative path
        rooted at the specified folder name.

        output format example.
        In the case of YOLO, the key is the tree structure
        if the tree structure is multi-level.

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
        yolo_annotations = self.__get_yolo_format_annotations(dataset_folder_path)

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
            id: class_name
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
                "size": [width, height]
            ...
        }
        """
        image_types = utils.get_supported_image_ext()
        image_paths = [
            p
            for p in glob.glob(
                os.path.join(dataset_folder_path, "**/*"), recursive=True
            )
            if re.search(r"/*\.({})".format("|".join(image_types)), str(p))
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
            annotation_file_path_without_ext:
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
        annotation_file_paths = [
            p
            for p in glob.glob(
                os.path.join(dataset_folder_path, "**/*.txt"), recursive=True
            )
            if re.search(r"/*\.txt", str(p))
        ]
        for annotation_file_path in annotation_file_paths:
            with open(annotation_file_path, "r") as f:
                anno_lines = f.readlines()
                annotation_key = annotation_file_path.replace(".txt", "")
                yolo_annotations[annotation_key] = []
                for anno_line in anno_lines:
                    yolo_annotations[annotation_key].append(
                        anno_line.strip().split(" ")
                    )
        return yolo_annotations

    # Task Convert

    def export_coco(
        self,
        project: str,
        tasks: list,
        annotations: list = [],
        output_dir: str = os.path.join("output", "coco"),
        output_file_name: str = "annotations.json",
    ) -> None:
        """
        Convert tasks to COCO format and export as a file.
        If you pass annotations, you can export Pose Estimation type annotations.

        project is slug of your project (Required).
        tasks is a list of tasks (Required).
        annotations is a list of annotations (Optional).
        output_dir is output directory(default: output/coco) (Optional).
        output_file_name is output file name(default: annotations.json) (Optional).
        """
        if not utils.is_json_ext(output_file_name):
            raise FastLabelInvalidException(
                "Output file name must have a json extension", 422
            )

        project = self.find_project_by_slug(project)
        if project is None:
            raise FastLabelInvalidException(
                "Project not found. Check the project slag.", 422
            )

        os.makedirs(output_dir, exist_ok=True)
        coco = converters.to_coco(
            project_type=project["type"],
            tasks=tasks,
            annotations=annotations,
            output_dir=output_dir,
        )
        file_path = os.path.join(output_dir, output_file_name)
        with open(file_path, "w") as f:
            json.dump(coco, f, indent=4, ensure_ascii=False)

    def export_yolo(
        self,
        project: str,
        tasks: list,
        classes: list = [],
        output_dir: str = os.path.join("output", "yolo"),
    ) -> None:
        """
        Convert tasks to YOLO format and export as files.
        If you pass classes, classes.txt will be generated based on it .
        If not , classes.txt will be generated based on passed tasks .
        (Annotations never used in your project will not be exported.)

        project is slug of your project (Required).
        tasks is a list of tasks (Required).
        classes is a list of annotation values.  e.g. ['dog','bird'] (Optional).
        output_dir is output directory(default: output/yolo) (Optional).
        """

        project = self.find_project_by_slug(project)
        if project is None:
            raise FastLabelInvalidException(
                "Project not found. Check the project slag.", 422
            )

        os.makedirs(output_dir, exist_ok=True)
        annos, categories = converters.to_yolo(
            project_type=project["type"],
            tasks=tasks,
            classes=classes,
            output_dir=output_dir,
        )
        for anno in annos:
            file_name = anno["filename"]
            basename = utils.get_basename(file_name)
            file_path = os.path.join(output_dir, "annotations", basename + ".txt")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf8") as f:
                objects = anno.get("object")
                if objects is None:
                    continue
                for obj in objects:
                    f.write(obj)
                    f.write("\n")
        classes_file_path = os.path.join(output_dir, "classes.txt")
        with open(classes_file_path, "w", encoding="utf8") as f:
            for category in categories:
                f.write(category["name"])
                f.write("\n")

    def export_pascalvoc(
        self,
        project: str,
        tasks: list,
        output_dir: str = os.path.join("output", "pascalvoc"),
    ) -> None:
        """
        Convert tasks to Pascal VOC format as files.

        project is slug of your project (Required).
        tasks is a list of tasks (Required).
        output_dir is output directory(default: output/pascalvoc) (Optional).
        """

        project = self.find_project_by_slug(project)
        if project is None:
            raise FastLabelInvalidException(
                "Project not found. Check the project slag.", 422
            )

        os.makedirs(output_dir, exist_ok=True)
        pascalvoc = converters.to_pascalvoc(
            project_type=project["type"], tasks=tasks, output_dir=output_dir
        )
        for voc in pascalvoc:
            file_name = voc["annotation"]["filename"]
            basename = utils.get_basename(file_name)
            file_path = os.path.join(output_dir, "annotations", basename + ".xml")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            xml = xmltodict.unparse(
                voc, pretty=True, indent="    ", full_document=False
            )
            with open(file_path, "w", encoding="utf8") as f:
                f.write(xml)

    def export_labelme(
        self, tasks: list, output_dir: str = os.path.join("output", "labelme")
    ) -> None:
        """
        Convert tasks to labelme format as files.

        tasks is a list of tasks (Required).
        output_dir is output directory(default: output/labelme) (Optional).
        """
        labelmes = converters.to_labelme(tasks)
        for labelme in labelmes:
            file_name = labelme["imagePath"]
            basename = utils.get_basename(file_name)
            file_path = os.path.join(output_dir, basename + ".json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(labelme, f, indent=4, ensure_ascii=False)

    # Instance / Semantic Segmetation

    def export_instance_segmentation(
        self,
        tasks: list,
        output_dir: str = os.path.join("output", "instance_segmentation"),
        pallete: List[int] = const.COLOR_PALETTE,
        start_index: int = 1,
    ) -> None:
        """
        Convert tasks to index color instance segmentation (PNG files).
        Supports only bbox, polygon and segmentation annotation types.
        Supports up to 57 instances in default colors palette.
        Check const.COLOR_PALETTE for more details.

        tasks is a list of tasks (Required).
        output_dir is output directory(default: output/instance_segmentation)(Optional).
        pallete is color palette of index color. Ex: [255, 0, 0, ...] (Optional).
        start_index is the first index of color index corresponding to color pallete.
            The expected values for start_index are either 0 or 1.
            When start_index is 0, all pixels are assumed to have annotations
            because they become the same color as the background(Optional).
        """
        tasks = converters.to_pixel_coordinates(tasks)
        for task in tasks:
            self.__export_index_color_image(
                task=task,
                output_dir=output_dir,
                pallete=pallete,
                is_instance_segmentation=True,
                start_index=start_index,
            )

    def export_semantic_segmentation(
        self,
        tasks: list,
        output_dir: str = os.path.join("output", "semantic_segmentation"),
        pallete: List[int] = const.COLOR_PALETTE,
        classes: List = [],
        start_index: int = 1,
    ) -> None:
        """
        Convert tasks to index color semantic segmentation (PNG files).
        Supports only bbox, polygon and segmentation annotation types.
        Check const.COLOR_PALETTE for color pallete.

        tasks is a list of tasks (Required).
        output_dir is output directory(default: output/semantic_segmentation)(Optional).
        pallete is color palette of index color. Ex: [255, 0, 0, ...] (Optional).
        classes is a list of annotation values.
            This list defines the value order that corresponds to the color index of the annotation.(Optional).
        start_index is the first index of color index corresponding to color pallete.
            The expected values for start_index are either 0 or 1.
            When start_index is 0, all pixels are assumed to have annotations
            because they become the same color as the background(Optional).
        """

        # Copy classes to target_classes
        # so that it is not added as a default argument.
        target_classes = classes.copy()
        if len(target_classes) == 0:
            for task in tasks:
                for annotation in task["annotations"]:
                    target_classes.append(annotation["value"])
            target_classes = list(set(target_classes))
            target_classes.sort()

        tasks = converters.to_pixel_coordinates(tasks)
        for task in tasks:
            self.__export_index_color_image(
                task=task,
                output_dir=output_dir,
                pallete=pallete,
                is_instance_segmentation=False,
                classes=target_classes,
                start_index=start_index,
            )

    def __export_index_color_image(
        self,
        task: list,
        output_dir: str,
        pallete: List[int],
        is_instance_segmentation: bool = True,
        classes: list = [],
        start_index: int = 1,
    ) -> None:
        image = Image.new("RGB", (task["width"], task["height"]), 0)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        index = start_index
        # In case segmentation, to avoid hollowed points overwrite other segmentation
        # in them, segmentation rendering process is different from
        # other annotation type
        seg_mask_images = []
        for annotation in task["annotations"]:
            color = (
                index
                if is_instance_segmentation
                else classes.index(annotation["value"]) + start_index
            )
            if annotation["type"] == AnnotationType.segmentation.value:
                # Create each annotation's masks and merge them finally
                seg_mask_ground = Image.new("RGB", (task["width"], task["height"]), 0)
                seg_mask_image = np.array(seg_mask_ground)
                seg_mask_image = cv2.cvtColor(seg_mask_image, cv2.COLOR_BGR2GRAY)

                for region in annotation["points"]:
                    count = 0
                    for points in region:
                        if count == 0:
                            cv_draw_points = []
                            if utils.is_clockwise(points):
                                cv_draw_points = self.__get_cv_draw_points(
                                    utils.sort_segmentation_points(points)
                                )
                            else:
                                reverse_points = utils.reverse_points(points)
                                sorted_points = utils.sort_segmentation_points(
                                    reverse_points
                                )
                                cv_draw_points = self.__get_cv_draw_points(
                                    sorted_points
                                )
                            cv2.fillPoly(
                                seg_mask_image,
                                [cv_draw_points],
                                color,
                                lineType=cv2.LINE_8,
                                shift=0,
                            )
                        else:
                            # Reverse hollow points for opencv because these points are
                            # counterclockwise
                            reverse_points = utils.reverse_points(points)
                            sorted_points = utils.sort_segmentation_points(
                                reverse_points
                            )
                            cv_draw_points = self.__get_cv_draw_points(sorted_points)
                            cv2.fillPoly(
                                seg_mask_image,
                                [cv_draw_points],
                                0,
                                lineType=cv2.LINE_8,
                                shift=0,
                            )
                        count += 1
                seg_mask_images.append(seg_mask_image)
            elif annotation["type"] == AnnotationType.polygon.value:
                cv_draw_points = self.__get_cv_draw_points(annotation["points"])
                cv2.fillPoly(
                    image, [cv_draw_points], color, lineType=cv2.LINE_8, shift=0
                )
            elif annotation["type"] == AnnotationType.bbox.value:
                cv_draw_points = self.__get_cv_draw_points(annotation["points"])
                cv2.fillPoly(
                    image, [cv_draw_points], color, lineType=cv2.LINE_8, shift=0
                )
            else:
                continue
            index += 1

        # For segmentation, merge each mask images
        for seg_mask_image in seg_mask_images:
            image = image | seg_mask_image

        image_path = os.path.join(output_dir, utils.get_basename(task["name"]) + ".png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image)
        image = image.convert("P")
        image.putpalette(pallete)
        image.save(image_path)

    def __get_cv_draw_points(self, points: List[int]) -> List[int]:
        """
        Convert points to pillow draw points. Diagonal points are not supported
        Annotation clockwise draw.
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

    def __reverse_points(self, points: List[int]) -> List[int]:
        """
        e.g.)
        [4, 5, 4, 9, 8, 9, 8, 5, 4, 5] => [4, 5, 8, 5, 8, 9, 4, 9, 4, 5]
        """
        reversed_points = []
        for index, _ in enumerate(points):
            if index % 2 == 0:
                reversed_points.insert(0, points[index + 1])
                reversed_points.insert(0, points[index])
        return reversed_points

    def __create_image_with_annotation(self, img_file_path_task):
        [img_file_path, task, output_dir] = img_file_path_task
        img = Image.open(img_file_path).convert("RGB")
        width, height = img.size
        if width > height:
            stroke_width = int(height / 300)
        else:
            stroke_width = int(width / 300)
        stroke_width = stroke_width if stroke_width > 1 else 1
        draw_img = ImageDraw.Draw(img, "RGBA")
        # For segmentation task
        is_seg = False
        seg_mask_images = []
        task_annotations = task["annotations"]
        for task_annotation in task_annotations:
            # Draw annotations in content
            rgb = None
            try:
                rgb = ImageColor.getcolor(task_annotation["color"], "RGB")
            except Exception as e:
                logger.info(e)
            if not rgb:
                continue
            rgba_dark = rgb + (OPACITY_DARK,)
            rgba_thin = rgb + (OPACITY_THIN,)
            if AnnotationType(task_annotation["type"]) == AnnotationType.bbox:
                points = task_annotation["points"]
                draw_img.rectangle(
                    points, fill=rgba_thin, outline=rgba_dark, width=stroke_width
                )
            elif AnnotationType(task_annotation["type"]) == AnnotationType.circle:
                x = task_annotation["points"][0]
                y = task_annotation["points"][1]
                radius = task_annotation["points"][2]
                points = [
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ]
                draw_img.ellipse(points, fill=rgba_dark, width=radius)
            elif AnnotationType(task_annotation["type"]) == AnnotationType.polygon:
                points = task_annotation["points"]
                # require start point at the end
                points.append(points[0])
                points.append(points[1])
                draw_img.line(points, fill=rgba_dark, width=stroke_width)
                draw_img.polygon(points, fill=rgba_thin)
            elif AnnotationType(task_annotation["type"]) == AnnotationType.keypoint:
                x = task_annotation["points"][0]
                y = task_annotation["points"][1]
                if stroke_width < KEYPOINT_MIN_STROKE_WIDTH:
                    stroke_width = KEYPOINT_MIN_STROKE_WIDTH
                points = [
                    x - stroke_width,
                    y - stroke_width,
                    x + stroke_width,
                    y + stroke_width,
                ]
                draw_img.ellipse(points, fill=rgba_dark, width=stroke_width)
            elif AnnotationType(task_annotation["type"]) == AnnotationType.line:
                points = task_annotation["points"]
                draw_img.line(points, fill=rgba_dark, width=stroke_width)
            elif AnnotationType(task_annotation["type"]) == AnnotationType.segmentation:
                is_seg = True
                rgba_seg = rgb + (OPACITY_THIN * 2,)
                seg_mask_ground = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                seg_mask_im = np.array(seg_mask_ground)
                for region in task_annotation["points"]:
                    count = 0
                    for points in region:
                        if count == 0:
                            cv_draw_points = self.__get_cv_draw_points(points)
                            # For diagonal segmentation points, fillPoly cannot rendering cv_draw_points, so convert
                            # shape. When sequential image project can use only pixel mode, remove it
                            converted_points = (
                                np.array(cv_draw_points)
                                .reshape((-1, 1, 2))
                                .astype(np.int32)
                            )
                            cv2.fillPoly(
                                seg_mask_im,
                                [converted_points],
                                rgba_seg,
                                lineType=cv2.LINE_8,
                                shift=0,
                            )
                        else:
                            # Reverse hollow points for opencv because this points are counter clockwise
                            cv_draw_points = self.__get_cv_draw_points(
                                self.__reverse_points(points)
                            )
                            converted_points = (
                                np.array(cv_draw_points)
                                .reshape((-1, 1, 2))
                                .astype(np.int32)
                            )
                            cv2.fillPoly(
                                seg_mask_im,
                                [converted_points],
                                (0, 0, 0, 0),
                                lineType=cv2.LINE_8,
                                shift=0,
                            )
                        count += 1
                seg_mask_images.append(seg_mask_im)
            elif (
                AnnotationType(task_annotation["type"])
                == AnnotationType.pose_estimation
            ):
                """
                {
                    keypoint_id: {
                        point: [x, y],
                        keypoint_rgb: keypoint.color
                    }
                }
                """
                if stroke_width < POSE_ESTIMATION_MIN_STROKE_WIDTH:
                    stroke_width = POSE_ESTIMATION_MIN_STROKE_WIDTH
                linked_points_and_color_to_key_map = {}
                relations = []
                for task_annotation_keypoint in task_annotation["keypoints"]:
                    try:
                        task_annotation_keypoint_keypoint_color = task_annotation[
                            "color"
                        ]
                        task_annotation_keypoint_name = task_annotation_keypoint["name"]
                        task_annotation_keypoint_value = task_annotation_keypoint[
                            "value"
                        ]
                        task_annotation_keypoint_key = task_annotation_keypoint["key"]
                        keypoint_rgb = ImageColor.getcolor(
                            task_annotation_keypoint_keypoint_color, "RGB"
                        )
                    except Exception as e:
                        logger.info(
                            f"Invalid color: {task_annotation_keypoint_keypoint_color}, "
                            f"content_name: {task_annotation_keypoint_name}, {e}"
                        )
                    if not keypoint_rgb:
                        continue
                    if not task_annotation_keypoint_value:
                        continue

                    x = task_annotation_keypoint_value[0]
                    y = task_annotation_keypoint_value[1]
                    linked_points_and_color_to_key_map[task_annotation_keypoint_key] = {
                        "point": [x, y],
                        "keypoint_rgb": keypoint_rgb,
                    }
                    for edge in task_annotation_keypoint["edges"]:
                        relations.append(
                            SEPARATOER.join(
                                sorted([task_annotation_keypoint_key, edge])
                            )
                        )

                for relation in set(relations):
                    first_key, second_key = relation.split(SEPARATOER)
                    if (
                        linked_points_and_color_to_key_map.get(first_key) is None
                        or linked_points_and_color_to_key_map.get(second_key) is None
                    ):
                        continue
                    line_start_point = linked_points_and_color_to_key_map.get(
                        first_key
                    )["point"]
                    line_end_point = linked_points_and_color_to_key_map.get(second_key)[
                        "point"
                    ]
                    relation_line_points = line_start_point + line_end_point

                    draw_img.line(
                        relation_line_points, fill=rgba_dark, width=stroke_width
                    )

                for key in linked_points_and_color_to_key_map:
                    x, y = linked_points_and_color_to_key_map[key]["point"]
                    points = [
                        x - stroke_width,
                        y - stroke_width,
                        x + stroke_width,
                        y + stroke_width,
                    ]
                    draw_img.ellipse(
                        points,
                        fill=linked_points_and_color_to_key_map[key]["keypoint_rgb"],
                        width=stroke_width,
                    )

        if is_seg:
            # For segmentation, merge each mask images with logical adding
            mask_seg_ground = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            mask_seg = np.array(mask_seg_ground)
            for seg_mask_image in seg_mask_images:
                mask_seg = mask_seg | seg_mask_image

            # Alpha brend original image and segmentation mask
            np_img = np.array(img.convert("RGBA"))
            merged_seg = np_img * 0.5 + mask_seg * 0.5
            # Composite all. 'merged_seg' will be used rendering annotation area,
            # other area will calcurate from 'mask_seg' and rendered by original image
            img = Image.composite(
                Image.fromarray(merged_seg.astype(np.uint8)),
                Image.fromarray(np_img.astype(np.uint8)),
                Image.fromarray(mask_seg.astype(np.uint8)),
            )

            # For export with original ext, if original image is not png foamat, convert RGB
            if os.path.splitext(img_file_path)[1].lower() != ".png":
                img = img.convert("RGB")
        # Save annotated content
        output_file_path = os.path.join(output_dir, task["name"])
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        img.save(output_file_path, quality=95)

    def export_image_with_annotations(
        self,
        tasks: list,
        image_dir: str,
        output_dir: str = os.path.join("output", "images_with_annotations"),
    ) -> None:
        """
        Export image with annotations
        """
        target_file_candidate_paths = glob.glob(
            os.path.join(image_dir, "**"), recursive=True
        )
        img_file_paths = []
        for target_file_candidate_path in target_file_candidate_paths:
            if not os.path.isfile(target_file_candidate_path):
                continue
            if not target_file_candidate_path.lower().endswith(
                EXPORT_IMAGE_WITH_ANNOTATIONS_SUPPORTED_IMAGE_TYPES
            ):
                continue
            img_file_paths.append(target_file_candidate_path)
        img_file_paths.sort()

        img_file_path_task_list = []
        for img_file_path in img_file_paths:
            slashed_img_file_path = img_file_path.replace(os.path.sep, "/")
            task_name = (
                slashed_img_file_path.replace(image_dir + "/", "")
                if not image_dir.endswith("/")
                else slashed_img_file_path.replace(image_dir, "")
            )
            task = next(
                filter(lambda x: x["name"] == task_name, tasks),
                None,
            )
            if not task:
                logger.info(f"Not find task. filepath: {task_name}")
                continue
            img_file_path_task_list.append([img_file_path, task, output_dir])

        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.__create_image_with_annotation, img_file_path_task_list)

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
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        value is a unique identifier of annotation in your project (Required).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
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
        attributes: list = [],
    ) -> str:
        """
        Create an annotation.

        project is slug of your project (Required).
        type can be 'bbox', 'polygon', 'keypoint', 'classification', 'line',
        'segmentation' (Required).
        value is a unique identifier of annotation in your project (Required).
        title is a display name of value (Required).
        color is hex color code like #ffffff (Optional).
        attributes is a list of attribute (Optional).
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

    def create_classification_annotation(self, project: str, attributes: list) -> str:
        """
        Create a classification annotation.

        project is slug of your project (Required).
        attributes is a list of attribute (Required).
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
        attributes: list = [],
    ) -> str:
        """
        Update an annotation.

        annotation_id is an id of the annotation (Required).
        value is a unique identifier of annotation in your project (Optional).
        title is a display name of value (Optional).
        color is hex color code like #ffffff (Optional).
        attributes is a list of attribute (Optional).
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
        self, annotation_id: str, attributes: list
    ) -> str:
        """
        Update a classification annotation.

        annotation_id is an id of the annotation (Required).
        attributes is a list of attribute (Required).
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

        slug is slug of your project (Required).
        """
        projects = self.get_projects(slug=slug)
        if not projects:
            return None
        return projects[0]

    def update_dicom_task(
        self,
        task_id: str,
        status: str = None,
        external_status: str = None,
        tags: list = [],
        **kwargs,
    ) -> str:
        """
        Update a single image task.

        task_id is an id of the task (Required).
        status can be 'registered', 'completed', 'skipped', 'reviewed', 'sent_back',
        'approved', 'declined' (Optional).
        external_status can be 'registered', 'completed', 'skipped', 'reviewed',
        'sent_back', 'approved', 'declined',  'customer_declined'. (Optional)
        tags is a list of tag to be set (Optional).
        assignee is slug of assigned user (Optional).
        reviewer is slug of review user (Optional).
        approver is slug of approve user (Optional).
        external_assignee is slug of external assigned user (Optional).
        external_reviewer is slug of external review user (Optional).
        external_approver is slug of external approve user (Optional).
        """
        endpoint = "tasks/dicom/" + task_id
        payload = {}
        if status:
            payload["status"] = status
        if external_status:
            payload["externalStatus"] = external_status
        if tags:
            payload["tags"] = tags

        self.__fill_assign_users(payload, **kwargs)

        return self.api.put_request(endpoint, payload=payload)

    def get_projects(
        self,
        slug: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of projects.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        slug is slug of your project (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
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
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
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

        type can be 'image_bbox', 'image_polygon', 'image_keypoint', 'image_line',
        'image_segmentation', 'image_classification', 'image_all', 'sequential_image_bbox',
        'sequential_image_polygon', 'sequential_image_keypoint', 'sequential_image_line',
        'sequential_image_segmentation', 'video_bbox',
        'video_single_classification' (Required).
        name is name of your project (Required).
        slug is slug of your project (Required).
        is_pixel is whether to annotate image with pixel level (Optional).
        job_size is the number of tasks the annotator gets at one time (Optional).
        workflow is the type of annotation workflow. workflow can be 'two_step' or
        'three_step' (Optional).
        external_workflow is the type of external annotation workflow. external_workflow
        can be 'two_step' or 'three_step' (Optional).
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

        project_id is an id of the project (Required).
        name is name of your project (Optional).
        slug is slug of your project (Optional).
        job_size is the number of tasks the annotator gets at one time (Optional).
        workflow is the type of annotation workflow. workflow can be 'two_step' or
        'three_step' (Optional).
        external_workflow is the type of external annotation workflow. external_workflow
        can be 'two_step' or 'three_step' (Optional).
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

    def copy_project(self, project_id: str) -> None:
        """
        Copy a project.
        """
        payload = {"id": project_id}
        endpoint = "projects/copy"
        return self.api.post_request(endpoint, payload=payload)

    # Tags

    def get_tags(
        self,
        project: str,
        keyword: str = None,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of tags.
        Returns up to 1000 at a time, to get more,
        project is slug of your project (Required).
        keyword are search terms in the tag name (Optional).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tags"
        params = {"project": project}
        if keyword:
            params["keyword"] = keyword
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.api.get_request(endpoint, params=params)

    def delete_tags(self, tag_ids: List[str]) -> None:
        """
        Delete a tags.
        """
        endpoint = "tags/delete/multi"
        payload = {"tagIds": tag_ids}
        self.api.post_request(endpoint, payload=payload)

    # Dataset

    def find_dataset(self, dataset_id: str) -> dict:
        """
        Find a dataset with latest version.
        """
        endpoint = "datasets-v2/" + dataset_id
        return self.api.get_request(endpoint)

    def get_datasets(
        self,
        keyword: str = None,
        tags: List[str] = [],
        license: Optional[str] = None,
        visibility: str = None,
    ) -> list:
        """
        Returns a list of datasets.

        keyword are search terms in the dataset slug (Optional).
        visibility are search terms in the dataset visibility.(Optional).
        """
        endpoint = "datasets-v2"
        params = {}
        if keyword:
            params["keyword"] = keyword
        if tags:
            params["tags"] = tags
        if license:
            params["license"] = license
        if visibility:
            params["visibility"] = visibility
        return self.api.get_request(endpoint, params=params)

    def create_dataset(
        self, name: str, tags: List[str] = [], visibility: str = None
    ) -> dict:
        """
        Create a dataset.

        name is name of your dataset. Only lowercase alphanumeric characters + hyphen is available (Required).
        tags is a list of tag (Optional).
        visibility are search terms in the dataset visibility.(Optional).
        """
        endpoint = "datasets-v2"
        payload = {"name": name}
        if tags:
            payload["tags"] = tags
        if visibility:
            payload["visibility"] = visibility
        return self.api.post_request(endpoint, payload=payload)

    def update_dataset(
        self,
        dataset_id: str,
        name: str = None,
        tags: List[str] = None,
    ) -> dict:
        """
        Update a dataset.

        dataset_id is an id of the dataset (Required).
        name is name of your dataset (Required).
        tags is a list of tag (Optional).
        """
        endpoint = "datasets-v2/" + dataset_id
        payload = {"name": name}
        if tags is not None:
            payload["tags"] = tags
        return self.api.put_request(endpoint, payload=payload)

    def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete a dataset.
        """
        endpoint = "datasets-v2/" + dataset_id
        self.api.delete_request(endpoint)

    # Dataset Object

    def find_dataset_object(
        self,
        dataset_id: str,
        object_name: str,
        version: str = None,
        revision_id: str = None,
    ) -> dict:
        """
        Find a dataset object.

        dataset_id is dataset id (Required).
        object_name is dataset object name (Required).
        version is dataset version (Optional).
        revision_id is dataset rebision (Optional).
        Only use specify one of revision_id or version.
        """
        if version and revision_id:
            raise FastLabelInvalidException(
                "only use specify one of revisionId or version.", 400
            )
        endpoint = "datasets-v2/" + dataset_id + "/objects/" + object_name
        params = {}
        if revision_id:
            params["revisionId"] = revision_id
        elif version:
            params["version"] = version
        return self.api.get_request(endpoint, params=params)

    def get_dataset_objects(
        self,
        dataset: str,
        version: str = None,
        tags: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
        revision_id: str = None,
        types: Optional[List[Union[str, DatasetObjectType]]] = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list:
        """
        Returns a list of dataset objects.

        dataset is dataset name (Required).
        version is dataset version (Optional).
        tags is a list of tag (Optional).
        revision_id is dataset rebision (Optional).
        Only use specify one of revision_id or version.
        """
        endpoint = "dataset-objects-v2"
        types = [DatasetObjectType.create(type_) for type_ in types or []]
        params = self._prepare_params(
            dataset=dataset,
            version=version,
            tags=tags,
            licenses=licenses,
            revision_id=revision_id,
            types=types,
            offset=offset,
            limit=limit,
        )
        return self.api.get_request(endpoint, params=params)

    def _prepare_params(
        self,
        dataset: str,
        offset: int,
        limit: int,
        version: str,
        revision_id: str,
        tags: Optional[List[str]],
        licenses: Optional[List[str]],
        types: Optional[List[DatasetObjectType]],
    ) -> DatasetObjectGetQuery:
        if version and revision_id:
            raise FastLabelInvalidException(
                "only use specify one of revisionId or version.", 400
            )
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        params: DatasetObjectGetQuery = {
            "dataset": dataset,
            "offset": offset,
            "limit": limit,
        }
        if revision_id:
            params["revisionId"] = revision_id
        if version:
            params["version"] = version
        if tags:
            params["tags"] = tags
        if licenses:
            params["licenses"] = licenses
        if types:
            params["types"] = [t.value for t in types]
        return params

    def download_dataset_objects(
        self,
        dataset: str,
        path: str,
        version: str = "",
        revision_id: str = "",
        tags: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
        types: Optional[List[Union[str, DatasetObjectType]]] = None,
        offset: int = 0,
        limit: int = 1000,
    ):
        endpoint = "dataset-objects-v2/signed-urls"
        types = [DatasetObjectType.create(type_) for type_ in types or []]
        params = self._prepare_params(
            dataset=dataset,
            offset=offset,
            limit=limit,
            version=version,
            revision_id=revision_id,
            tags=tags,
            types=types,
            licenses=licenses,
        )

        response = self.api.get_request(endpoint, params=params)

        download_path = Path(path)
        download_path.mkdir(exist_ok=True)
        object_map = {}
        if types:
            for type_ in types:
                (download_path / type_.value).mkdir(exist_ok=True)
                object_map[type_.value] = [
                    obj for obj in response if obj["type"] == type_.value
                ]
        else:
            object_map[""] = response

        for _type, objects in object_map.items():
            base_path = download_path / _type
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.__download_dataset_object, base_path, obj)
                    for obj in objects
                ]
                wait(futures)

            # check specification
            output_path = base_path / "annotations.json"
            exist_dataset_objects = []
            if os.path.exists(output_path):
                exist_dataset_objects = json.load(open(output_path))
            with Path(base_path / "annotations.json").open("w") as f:
                annotations = [
                    {
                        "name": obj["name"],
                        "annotations": obj["annotations"],
                        "customMetadata": obj["customMetadata"],
                        "tags": obj["tags"],
                        "object_type": obj["type"],
                    }
                    for obj in objects
                ]
                json.dump(
                    exist_dataset_objects + annotations,
                    fp=f,
                    ensure_ascii=False,
                    indent=4,
                )
        return [obj for objects in object_map.values() for obj in objects]

    def __download_dataset_object(self, download_path: Path, obj: dict):
        obj_path = download_path / obj["name"]
        os.makedirs(obj_path.parent, exist_ok=True)
        response = requests.get(obj["signedUrl"])
        with obj_path.open("wb") as f:
            f.write(response.content)

    def create_dataset_object(
        self,
        dataset: str,
        name: str,
        file_path: str,
        tags: List[str] = None,
        licenses: List[str] = None,
        annotations: List[dict] = None,
        custom_metadata: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Create a dataset object.

        dataset is dataset name (Required).
        name is a unique identifier of dataset object in your dataset (Required).
        file_path is a path to data. (Required).
        tags is a list of tag (Optional).
        annotations is a list of annotation (Optional).
        """
        tags = tags or []
        annotations = annotations or []
        endpoint = "dataset-objects-v2"
        if not utils.is_object_supported_size(file_path):
            raise FastLabelInvalidException(
                "Supported object size is under 250 MB.", 422
            )
        payload = {
            "dataset": dataset,
            "name": name,
            "filePath": utils.base64_encode(file_path),
        }
        if tags:
            payload["tags"] = tags
        if licenses:
            payload["licenses"] = licenses
        if annotations:
            payload["annotations"] = annotations
        if custom_metadata:
            payload["customMetadata"] = custom_metadata
        return self.api.post_request(endpoint, payload=payload)

    def update_dataset_object(
        self,
        dataset_id: str,
        object_name: str,
        tags: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
        annotations: Optional[List[dict]] = None,
        custom_metadata: Optional[dict] = None,
    ) -> dict:
        endpoint = "dataset-objects-v2"
        payload = {"datasetId": dataset_id, "objectName": object_name}
        if tags is not None:
            payload["tags"] = tags
        if licenses is not None:
            payload["licenses"] = licenses
        if annotations is not None:
            payload["annotations"] = annotations
        if custom_metadata is not None:
            payload["customMetadata"] = custom_metadata
        return self.api.put_request(endpoint, payload=payload)

    def delete_dataset_object(self, dataset_id: str, object_name: str) -> None:
        """
        Delete a dataset object.
        """
        endpoint = "datasets-v2/" + dataset_id + "/objects/" + object_name
        self.api.delete_request(endpoint)

    def update_aws_s3_storage(
        self, project: str, bucket_name: str, bucket_region: str, prefix: str = None
    ) -> str:
        """
        Insert or update AWS S3 storage settings.

        project is a slug of the project (Required).
        bucket_name is a bucket name of the aws s3 (Required).
        bucket_region is a bucket region of the aws s3 (Required).
        prefix is a folder name in the aws s3 bucket. (Optional).
        If sample_dir is specified as a prefix in the case of a hierarchical structure like the bucket below,
        only the data under the sample_dir directory will be linked.
        If not specified, everything under the bucket will be linked.

        [tree structure]
        fastlabel
        ├── sample1.jpg
        └── sample_dir
            └── sample2.jpg

        """
        endpoint = "storages/aws-s3/" + project
        payload = {
            "bucketName": bucket_name,
            "bucketRegion": bucket_region,
        }
        if prefix:
            payload["prefix"] = prefix
        return self.api.put_request(endpoint, payload=payload)

    def create_task_from_aws_s3(
        self,
        project: str,
        status: str = "registered",
        tags: List[str] = [],
        priority: int = 0,
    ) -> dict:
        """
        Insert or update AWS S3 storage settings.

        project is a slug of the project (Required).
        status can be 'registered', 'completed', 'skipped',
        'reviewed', 'sent_back', 'approved', 'declined' (default: registered) (Optional).
        tags is a list of tag (default: []) (Optional).
        priority is the priority of the task (default: none) (Optional).
        Set one of the numbers corresponding to:
            none = 0,
            low = 10,
            medium = 20,
            high = 30,
        """
        endpoint = "tasks/aws-s3"
        payload = {
            "project": project,
            "status": status,
            "tags": tags,
            "priority": priority,
        }
        return self.api.post_request(endpoint, payload=payload)

    def get_aws_s3_import_status_by_project(
        self,
        project: str,
    ) -> dict:
        """
        Returns a import status of create task from AWS S3.
        """
        endpoint = "tasks/import/status/aws-s3/" + project
        return self.api.get_request(endpoint)

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
        if "workflow_1_user" in kwargs:
            payload["workflow1User"] = kwargs.get("workflow_1_user")
        if "workflow_2_user" in kwargs:
            payload["workflow2User"] = kwargs.get("workflow_2_user")
        if "workflow_3_user" in kwargs:
            payload["workflow3User"] = kwargs.get("workflow_3_user")
        if "workflow_4_user" in kwargs:
            payload["workflow4User"] = kwargs.get("workflow_4_user")
        if "workflow_5_user" in kwargs:
            payload["workflow5User"] = kwargs.get("workflow_5_user")
        if "workflow_6_user" in kwargs:
            payload["workflow6User"] = kwargs.get("workflow_6_user")

    def __get_signed_path(
        self,
        project: str,
        file_name: str,
        file_type: str,
    ):
        endpoint = "files"
        params = {"project": project, "fileName": file_name, "fileType": file_type}
        return self.api.get_request(endpoint, params)

    def get_training_jobs(
        self,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of training jobs.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "trainings"
        params = {}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit

        return self.api.get_request(endpoint, params=params)

    def execute_training_job(
        self,
        dataset_name: str,
        base_model_name: str,
        epoch: int,
        dataset_revision_id: str = None,
        use_dataset_train_val: bool = False,
        instance_type: str = "ml.p3.2xlarge",
        batch_size: int = None,
        learning_rate: float = None,
        resize_option: Optional[Literal["fixed", "none"]] = None,
        resize_dimension: Optional[int] = None,
        annotation_value: str = "",
        config_file_path: Optional[Union[Path, str]] = None,
    ) -> list:
        """
        Returns a list of training jobs.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        endpoint = "trainings"
        payload = {
            "datasetName": dataset_name,
            "baseModelName": base_model_name,
            "epoch": epoch,
            "useDatasetTrainVal": use_dataset_train_val,
            "datasetRevisionId": dataset_revision_id,
            "instanceType": instance_type,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "resizeOption": resize_option,
            "resizeDimension": resize_dimension,
            "configFile": utils.base64_encode(str(config_file_path))
            if config_file_path is not None
            else None,
        }
        if annotation_value:
            payload["annotationValue"] = annotation_value

        return self.api.post_request(
            endpoint,
            payload={key: value for key, value in payload.items() if value is not None},
        )

    def find_training_job(self, id: str) -> list:
        """
        Returns training job.
        id is id of training job (Required).
        """

        endpoint = f"trainings/{id}"

        return self.api.get_request(
            endpoint,
        )

    def get_evaluation_jobs(
        self,
        offset: int = None,
        limit: int = 1000,
    ) -> list:
        """
        Returns a list of training jobs.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "evaluations"
        params = {}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit

        return self.api.get_request(endpoint, params=params)

    def find_evaluation_job(self, id: str) -> list:
        """
        Returns evaluation job.
        id is id of evaluation job (Required).
        """

        endpoint = f"evaluations/{id}"

        return self.api.get_request(
            endpoint,
        )

    def execute_evaluation_job(
        self,
        dataset_name: str,
        model_name: str,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.4,
        dataset_revision_id: str = None,
        use_dataset_test: bool = False,
        instance_type: str = "ml.p3.2xlarge",
        batch_size: int = None,
        learning_rate: float = None,
    ) -> list:
        """
        Returns a list of training jobs.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        endpoint = "evaluations"
        payload = {
            "modelName": model_name,
            "datasetName": dataset_name,
            "iouThreshold": iou_threshold,
            "confidenceThreshold": confidence_threshold,
            "datasetRevisionId": dataset_revision_id,
            "useDatasetTest": use_dataset_test,
        }

        return self.api.post_request(
            endpoint,
            payload={key: value for key, value in payload.items() if value is not None},
        )

    def execute_endpoint(
        self,
        endpoint_name: str,
        file_path: str,
    ) -> dict:
        # """
        # Execute existing endpoint.
        # endpoint_name is name of target endpoint (Required).
        # file_path is a path to data.
        # Supported extensions are png, jpg, jpeg (Required).
        # """
        endpoint = "model-endpoints/execute"
        if not utils.is_image_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422
            )
        if not utils.is_image_supported_size_for_inference(file_path):
            raise FastLabelInvalidException("Supported image size is under 6 MB.", 422)

        payload = {
            "modelEndpointName": endpoint_name,
            "file": utils.base64_encode(file_path),
        }
        return self.api.post_request(endpoint, payload=payload)

    def create_model_monitoring_request_results(
        self,
        name: str,
        results: list = [],
    ) -> str:
        """
        Create model monitoring request results.
        name is an unique identifier of model monitoring setting in your workspace (Required).
        results is a list of request result to be set (Required).
        """
        endpoint = "model-monitorings/create"

        payload = {"name": name, "results": results}

        return self.api.post_request(endpoint, payload=payload)

    def get_histories(
        self,
        project: str,
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of histories.
        Returns up to 1000 at a time, to get more, set offset as the starting position
        to fetch.

        project is slug of your project (Required).
        offset is the starting position number to fetch (Optional).
        limit is the max number to fetch (Optional).
        """
        if limit > 1000:
            raise FastLabelInvalidException(
                "Limit must be less than or equal to 1000.", 422
            )
        endpoint = "tasks/import/histories"
        params = {"project": project}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit

        return self.api.get_request(endpoint, params=params)

    def mask_to_fastlabel_segmentation_points(
        self, mask_image: Union[str, np.ndarray]
    ) -> List[List[List[int]]]:
        return [
            [converters.get_pixel_coordinates(p) for p in point]
            for point in utils.mask_to_segmentation(mask_image)
        ]


def delete_extra_annotations_parameter(annotations: list) -> list:
    for annotation in annotations:
        annotation.pop("id", None)
        annotation.pop("title", None)
        annotation.pop("color", None)
        for keypoint in annotation.get("keypoints", []):
            keypoint.pop("edges", None)
            keypoint.pop("name", None)
        annotation["attributes"] = delete_extra_attributes_parameter(
            annotation.get("attributes", [])
        )
    return annotations


def delete_extra_attributes_parameter(attributes: list) -> list:
    for attribute in attributes:
        attribute.pop("title", None)
        attribute.pop("name", None)
        attribute.pop("type", None)
    return attributes
