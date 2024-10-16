import os
from typing import Union

import requests

from .exceptions import FastLabelException, FastLabelInvalidException


class Api:
    base_url = "https://api.fastlabel.ai/v1/"

    access_token = None

    def __init__(self):
        if os.environ.get("FASTLABEL_API_URL"):
            self.base_url = os.environ.get("FASTLABEL_API_URL")
        if not os.environ.get("FASTLABEL_ACCESS_TOKEN"):
            raise ValueError("FASTLABEL_ACCESS_TOKEN is not configured.")
        self.access_token = "Bearer " + os.environ.get("FASTLABEL_ACCESS_TOKEN")

    def get_request(self, endpoint: str, params=None) -> Union[dict, list]:
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
        r = requests.get(self.base_url + endpoint, headers=headers, params=params)

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

    def delete_request(self, endpoint: str, params=None) -> dict:
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
        r = requests.delete(self.base_url + endpoint, headers=headers, params=params)

        if r.status_code == 200 or r.status_code == 204:
            return

        try:
            error = r.json()["message"]
        except ValueError:
            error = r.text
        if str(r.status_code).startswith("4"):
            raise FastLabelInvalidException(error, r.status_code)
        else:
            raise FastLabelException(error, r.status_code)

    def post_request(self, endpoint, payload=None):
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
        r = requests.post(self.base_url + endpoint, json=payload, headers=headers)

        if r.status_code == 200:
            return r.json()
        elif r.status_code == 204:
            return
        else:
            try:
                error = r.json()["message"]
            except ValueError:
                error = r.text
            if str(r.status_code).startswith("4"):
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def put_request(self, endpoint, payload=None):
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
        r = requests.put(self.base_url + endpoint, json=payload, headers=headers)

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

    def upload_zipfile(
        self,
        url: str,
        file_path: str,
    ):
        files = {"file": open(file_path, "rb")}
        return requests.put(url, files=files)
