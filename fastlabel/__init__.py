import os
from logging import getLogger

import requests

from fastlabel.const import AnalysisType

logger = getLogger(__name__)

APP_BASE_URL = "https://app.fastlabel.ai/projects/"
FASTLABEL_ENDPOINT = "https://api-fastlabel-production.web.app/api/v1/"


class Client:

    api_key = None

    def __init__(self) -> None:
        if not os.environ.get("FASTLABEL_API_KEY"):
            raise ValueError("FASTLABEL_API_KEY is not configured.")
        self.api_key = "Bearer " + os.environ.get("FASTLABEL_API_KEY")

    def _getrequest(self, endpoint: str, params=None) -> dict:
        """Makes a get request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'statusCode': XXX,
              'error': 'I failed' }
        """
        params = params or {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }
        r = requests.get(FASTLABEL_ENDPOINT + endpoint, headers=headers, params=params)

        if r.status_code == 200:
            return r.json()
        else:
            try:
                print(r.json())
                error = r.json()["error"]
            except ValueError:
                error = r.text
            if r.status_code == 400:
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
            "Authorization": self.api_key,
        }
        r = requests.delete(
            FASTLABEL_ENDPOINT + endpoint, headers=headers, params=params
        )

        if r.status_code == 200:
            return r.json()
        else:
            try:
                error = r.json()["error"]
            except ValueError:
                error = r.text
            if r.status_code == 400:
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
            "Authorization": self.api_key,
        }
        r = requests.post(FASTLABEL_ENDPOINT + endpoint, json=payload, headers=headers)

        if r.status_code == 200:
            return r.json()
        else:
            try:
                error = r.json()["error"]
            except ValueError:
                error = r.text
            if r.status_code == 400:
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def upload_predictions(
        self,
        project_id: str,
        analysis_type: AnalysisType,
        threshold: int,
        predictions: list,
    ) -> None:
        """
        Upload predictions to analyze your model.
        """
        endpoint = "predictions/upload"
        payload = {
            "projectId": project_id,
            "analysisType": analysis_type,
            "threshold": threshold,
            "predictions": predictions,
        }
        self._postrequest(endpoint, payload=payload)
        logger.warn(
            "Successfully uploaded! See " + APP_BASE_URL + project_id + "/modelAnalysis"
        )

    def find_task(self, task_id: str) -> dict:
        """
        Find a signle task.
        """
        endpoint = "tasks/" + task_id
        return self._getrequest(endpoint)

    def get_tasks(
        self,
        project_id: str,
        status: str = None,
        review_status: str = None,
        limit: int = 100,
        start_after: str = None,
    ) -> dict:
        """
        Returns a list of tasks.

        Returns up to 100 at a time, to get more, set task id of the last page passed back to startAfter param.

        project_id is id of your project. (Required)
        status can be 'registered', 'registered', 'submitted' or 'skipped'. (Optional)
        review_status can be 'notReviewed', 'inProgress', 'accepted' or 'declined'. (Optional)
        limit is the max number of results to display per page, (Optional)
        start_after can be use to fetch the next page of tasks. (Optional)
        """
        endpoint = "tasks/"
        params = {"projectId": project_id}
        if status:
            params["status"] = status
        if review_status:
            params["reviewStatus"] = review_status
        if limit:
            params["limit"] = limit
        if start_after:
            params["startAfter"] = start_after
        return self._getrequest(endpoint, params=params)

    def delete_task(self, task_id: str) -> None:
        """
        Delete a single task.
        """
        endpoint = "tasks/" + task_id
        self._deleterequest(endpoint)

    def create_image_task(
        self,
        project_id: str,
        key: str,
        url: str,
        status: str = None,
        review_status: str = None,
        labels: list = [],
    ) -> dict:
        """
        Create a single task for image project.

        project_id is id of your project.
        key is an unique identifier of task in your project. (Required)
        url is a link to get image data. (Required)
        status can be 'registered', 'inProgress', 'submitted' or 'skipped'. (Optional)
        review_status can be 'notReviewed', 'inProgress', 'accepted' or 'declined'. (Optional)
        labels is a list of label to be set in advance. (Optional)
        """
        endpoint = "tasks/image"
        payload = {"projectId": project_id, "key": key, "url": url}
        if status:
            payload["status"] = status
        if review_status:
            payload["review_status"] = review_status
        if labels:
            payload["labels"] = labels
        return self._postrequest(endpoint, payload=payload)


class FastLabelException(Exception):
    def __init__(self, message, errcode):
        super(FastLabelException, self).__init__(
            "<Response [{}]> {}".format(errcode, message)
        )
        self.code = errcode


class FastLabelInvalidException(FastLabelException, ValueError):
    pass
