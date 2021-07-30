import os
import requests

from .exceptions import FastLabelException, FastLabelInvalidException

FASTLABEL_ENDPOINT = "https://api.fastlabel.ai/v1/"


class Api:

    access_token = None

    def __init__(self):
        if not os.environ.get("FASTLABEL_ACCESS_TOKEN"):
            raise ValueError("FASTLABEL_ACCESS_TOKEN is not configured.")
        self.access_token = "Bearer " + \
            os.environ.get("FASTLABEL_ACCESS_TOKEN")

    def get_request(self, endpoint: str, params=None) -> dict:
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
