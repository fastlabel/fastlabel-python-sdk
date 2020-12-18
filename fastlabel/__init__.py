import logging
import os
import requests
from fastlabel.const import AnalysisType

FASTLABEL_ENDPOINT = "https://api-fastlabel-production.web.app/api/v1/"

class Client:

    api_key = None

    def __init__(self) -> None:
        if not os.environ.get('FASTLABEL_API_KEY'):
            raise ValueError("FASTLABEL_API_KEY is not configured.")
        self.api_key = "Bearer " + os.environ.get('FASTLABEL_API_KEY')

    def _getrequest(self, endpoint: str, params=None) -> dict:
        """Makes a get request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'status_code': XXX,
              'error': 'I failed' }
        """
        params = params or {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }
        r = requests.get(FASTLABEL_ENDPOINT + endpoint,
                         headers=headers, params=params)

        if r.status_code == 200:
            return r.json()
        else:
            try:
                error = r.json()['error']
            except ValueError:
                error = r.text
            if r.status_code == 400:
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)
    
    def _postrequest(self, endpoint, payload=None):
        """Makes a post request to an endpoint.
        If an error occurs, assumes that endpoint returns JSON as:
            { 'status_code': XXX,
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
                error = r.json()['error']
            except ValueError:
                error = r.text
            if r.status_code == 400:
                raise FastLabelInvalidException(error, r.status_code)
            else:
                raise FastLabelException(error, r.status_code)

    def upload_predictions(self, project_id: str, analysis_type: AnalysisType, threshold: int, predictions: list) -> None:
        endpoint = "predictions/upload"
        payload = {
            "projectId": project_id,
            "analysisType": analysis_type,
            "threshold": threshold,
            "predictions": predictions
        }
        self._postrequest(endpoint, payload=payload)
        logging.info("")


class FastLabelException(Exception):
    def __init__(self, message, errcode):
        super(FastLabelException, self).__init__(
            '<Response [{}]> {}'.format(errcode, message))
        self.code = errcode


class FastLabelInvalidException(FastLabelException, ValueError):
    pass