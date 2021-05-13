import os
import glob
from logging import getLogger

import requests
import base64

logger = getLogger(__name__)

FASTLABEL_ENDPOINT = "https://api.fastlabel.ai/v1/"


class Client:

    access_token = None

    def __init__(self) -> None:
        if not os.environ.get("FASTLABEL_ACCESS_TOKEN"):
            raise ValueError("FASTLABEL_ACCESS_TOKEN is not configured.")
        self.access_token = "Bearer " + \
            os.environ.get("FASTLABEL_ACCESS_TOKEN")

    def _getrequest(self, endpoint: str, params=None) -> dict:
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

    def _deleterequest(self, endpoint: str, params=None) -> dict:
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

    def _postrequest(self, endpoint, payload=None):
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

    def _putrequest(self, endpoint, payload=None):
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
        return self._getrequest(endpoint)

    def find_multi_image_task(self, task_id: str) -> dict:
        """
        Find a signle multi image task.
        """
        endpoint = "tasks/multi/image/" + task_id
        return self._getrequest(endpoint)

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
        return self._getrequest(endpoint, params=params)

    def get_multi_image_tasks(
        self,
        project: str,
        status: str = None,
        tags: list = [],
        offset: int = None,
        limit: int = 100,
    ) -> dict:
        """
        Returns a list of tasks.
        Returns up to 1000 at a time, to get more, set offset as the starting position to fetch.

        project is slug of your project. (Required)
        status can be 'registered', 'in_progress', 'completed', 'skipped', 'in_review', 'send_backed', 'approved', 'customer_in_review', 'customer_send_backed', 'customer_approved'. (Optional)
        tags is a list of tag. (Optional)
        offset is the starting position number to fetch. (Optional)
        limit is the max number to fetch. (Optional)
        """
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
        return self._getrequest(endpoint, params=params)

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
        return self._postrequest(endpoint, payload=payload)
    
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
        return self._postrequest(endpoint, payload=payload)

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
        return self._putrequest(endpoint, payload=payload)

    def delete_task(self, task_id: str) -> None:
        """
        Delete a single task.
        """
        endpoint = "tasks/" + task_id
        self._deleterequest(endpoint)

    def __base64_encode(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def __is_supported_ext(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))


class FastLabelException(Exception):
    def __init__(self, message, errcode):
        super(FastLabelException, self).__init__(
            "<Response [{}]> {}".format(errcode, message)
        )
        self.code = errcode


class FastLabelInvalidException(FastLabelException, ValueError):
    pass
