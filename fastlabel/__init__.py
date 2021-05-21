import os
import glob
from enum import Enum
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor

import requests
import base64
import numpy as np
import geojson

logger = getLogger(__name__)

FASTLABEL_ENDPOINT = "https://api.fastlabel.ai/v1/"


class Client:

    access_token = None

    def __init__(self) -> None:
        if not os.environ.get("FASTLABEL_ACCESS_TOKEN"):
            raise ValueError("FASTLABEL_ACCESS_TOKEN is not configured.")
        self.access_token = "Bearer " + \
            os.environ.get("FASTLABEL_ACCESS_TOKEN")

    def __getrequest(self, endpoint: str, params=None) -> dict:
        """Makes a get request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'statusCode': XXX,
              'error': 'I failed' }
        """
        params = params or {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token,
        }
        r = requests.get(FASTLABEL_ENDPOINT + endpoint,
                         headers=headers, params=params)

        if r.status_code == 200:
            return r.json()
        else:
            try:
                error = r.json()["message"]
            except ValueError:
                error = r.text
            if str(r.status_code).startswith("4"):
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def __deleterequest(self, endpoint: str, params=None) -> dict:
        """Makes a delete request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'statusCode': XXX,
              'error': 'I failed' }
        """
        params = params or {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token,
        }
        r = requests.delete(
            FASTLABEL_ENDPOINT + endpoint, headers=headers, params=params
        )

        if r.status_code != 204:
            try:
                error = r.json()["message"]
            except ValueError:
                error = r.text
            if str(r.status_code).startswith("4"):
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def __postrequest(self, endpoint, payload=None):
        """Makes a post request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'statusCode': XXX,
              'error': 'I failed' }
        """
        payload = payload or {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token,
        }
        r = requests.post(FASTLABEL_ENDPOINT + endpoint,
                          json=payload, headers=headers)

        if r.status_code == 200:
            return r.json()
        else:
            try:
                error = r.json()["message"]
            except ValueError:
                error = r.text
            if str(r.status_code).startswith("4"):
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def __putrequest(self, endpoint, payload=None):
        """Makes a put request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'statusCode': XXX,
              'error': 'I failed' }
        """
        payload = payload or {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token,
        }
        r = requests.put(FASTLABEL_ENDPOINT + endpoint,
                         json=payload, headers=headers)

        if r.status_code == 200:
            return r.json()
        else:
            try:
                error = r.json()["message"]
            except ValueError:
                error = r.text
            if str(r.status_code).startswith("4"):
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def find_task(self, task_id: str) -> dict:
        """
        Find a signle task.
        """
        endpoint = "tasks/" + task_id
        return self.__getrequest(endpoint)

    def find_multi_image_task(self, task_id: str) -> dict:
        """
        Find a signle multi image task.
        """
        endpoint = "tasks/multi/image/" + task_id
        return self.__getrequest(endpoint)

    def get_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        offset: int = None,
        limit: int = 100,
    ) -> list:
        """
        Returns a list of tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
        endpoint = "tasks"
        params = {"project": project}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.__getrequest(endpoint, params=params)

    def get_multi_image_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        offset: int = None,
        limit: int = 10,
    ) -> dict:
        """
        Returns a list of tasks.
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
        endpoint = "tasks/multi/image"
        params = {"project": project}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return self.__getrequest(endpoint, params=params)

    def create_task(
        self,
        project: str,
        name: str,
        file_path: str,
        status: str = None,
        annotations: list = [],
        tags: list = [],
    ) -> str:
        """
        Create a single task.

        project is slug of your project. (Required)
        name is an unique identifier of task in your project. (Required)
        file_path is a path to data. Supported extensions are png, jpg, jpeg. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        annotations is a list of annotation to be set in advance. (Optional)
        tags is a list of tag to be set in advance. (Optional)
        """
        endpoint = "tasks"
        if not self.__is_supported_ext(file_path):
            raise FastLabelInvalidException(
                "Supported extensions are png, jpg, jpeg.", 422)
        file = self.__base64_encode(file_path)
        payload = {"project": project, "name": name, "file": file}
        if status:
            payload["status"] = status
        if annotations:
            for annotation in annotations:
                annotation["content"] = name
            payload["annotations"] = annotations
        if tags:
            payload["tags"] = tags
        return self.__postrequest(endpoint, payload=payload)

    def create_multi_image_task(
        self,
        project: str,
        name: str,
        folder_path: str,
        status: str = None,
        annotations: list = [],
        tags: list = [],
    ) -> dict:
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

        endpoint = "tasks/multi/image"
        file_paths = glob.glob(os.path.join(folder_path, "*"))
        contents = []
        for file_path in file_paths:
            if not self.__is_supported_ext(file_path):
                raise FastLabelInvalidException(
                    "Supported extensions are png, jpg, jpeg.", 422)
            file = self.__base64_encode(file_path)
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
        return self.__postrequest(endpoint, payload=payload)

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
        return self.__putrequest(endpoint, payload=payload)

    def delete_task(self, task_id: str) -> None:
        """
        Delete a single task.
        """
        endpoint = "tasks/" + task_id
        self.__deleterequest(endpoint)

    def to_coco(self, tasks: list) -> dict:
        # Get categories
        categories = self.__get_categories(tasks)

        # Get images and annotations
        images = []
        annotations = []
        annotation_id = 0
        image_id = 0
        for task in tasks:
            if task["height"] == 0 or task["width"] == 0:
                continue

            image_id += 1
            image = {
                "file_name": task["name"],
                "height": task["height"],
                "width": task["width"],
                "id": image_id,
            }
            images.append(image)

            data = [{"annotation": annotation, "categories": categories,
                     "image": image} for annotation in task["annotations"]]
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = executor.map(self.__to_annotation, data)

            for result in results:
                annotation_id += 1
                if not result:
                    continue
                result["id"] = annotation_id
                annotations.append(result)

        return {
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }

    def __base64_encode(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def __is_supported_ext(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))

    def __get_categories(self, tasks: list) -> list:
        values = []
        for task in tasks:
            for annotation in task["annotations"]:
                if annotation["type"] != AnnotationType.bbox.value and annotation["type"] != AnnotationType.polygon.value:
                    continue
                values.append(annotation["value"])
        values = list(set(values))

        categories = []
        for index, value in enumerate(values):
            category = {
                "supercategory": value,
                "id": index + 1,
                "name": value
            }
            categories.append(category)
        return categories

    def __to_annotation(self, data: dict) -> dict:
        annotation = data["annotation"]
        categories = data["categories"]
        image = data["image"]
        points = annotation["points"]
        annotation_type = annotation["type"]
        annotation_id = 0

        if annotation_type != AnnotationType.bbox.value and annotation_type != AnnotationType.polygon.value:
            return None
        if not points or len(points) == 0:
            return None
        if annotation_type == AnnotationType.bbox.value and (int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])):
            return None

        category = self.__get_category_by_name(categories, annotation["value"])

        return self.__get_annotation(
            annotation_id, points, category["id"], image, annotation_type)

    def __get_category_by_name(self, categories: list, name: str) -> str:
        category = [
            category for category in categories if category["name"] == name][0]
        return category

    def __get_annotation(self, id_: int, points: list, category_id: int, image: dict, annotation_type: str) -> dict:
        annotation = {}
        annotation["segmentation"] = [points]
        annotation["iscrowd"] = 0
        annotation["area"] = self.__calc_area(annotation_type, points)
        annotation["image_id"] = image["id"]
        annotation["bbox"] = self.__to_bbox(points)
        annotation["category_id"] = category_id
        annotation["id"] = id_
        return annotation

    def __to_bbox(self, points: list) -> list:
        points_splitted = [points[idx:idx + 2]
                           for idx in range(0, len(points), 2)]
        polygon_geo = geojson.Polygon(points_splitted)
        coords = np.array(list(geojson.utils.coords(polygon_geo)))
        left_top_x = coords[:, 0].min()
        left_top_y = coords[:, 1].min()
        right_bottom_x = coords[:, 0].max()
        right_bottom_y = coords[:, 1].max()

        return [
            left_top_x,  # x
            left_top_y,  # y
            right_bottom_x - left_top_x,  # width
            right_bottom_y - left_top_y,  # height
        ]

    def __calc_area(self, annotation_type: str, points: list) -> float:
        area = 0
        if annotation_type == AnnotationType.bbox.value:
            width = points[0] - points[2]
            height = points[1] - points[3]
            area = width * height
        elif annotation_type == AnnotationType.polygon.value:
            x = points[0::2]
            y = points[1::2]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                                np.dot(y, np.roll(x, 1)))
        return area


class AnnotationType(Enum):
    bbox = "bbox"
    polygon = "polygon"
    keypoint = "keypoint"
    classification = "classification"
    line = "line"
    segmentation = "segmentation"


class FastLabelException(Exception):
    def __init__(self, message, errcode):
        super(FastLabelException, self).__init__(
            "<Response [{}]> {}".format(errcode, message)
        )
        self.code = errcode


class FastLabelInvalidException(FastLabelException, ValueError):
    pass
