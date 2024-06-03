import os
from dotenv import load_dotenv
import requests
from .endpoints.regression import RegressionAPI
from .endpoints.datasource import DatasourceAPI
from .endpoints.classification import ClassificationAPI
from .endpoints.clustering import ClusteringAPI
from .endpoints.workflows import WorkflowsAPI

load_dotenv() 

class PredictEasyClient:
    def __init__(self, auth_key, auth_secret):
        self.auth_key = auth_key
        self.auth_secret = auth_secret
        self.account_services = os.getenv('ACCOUNT_SERVICES')
        self.timeseries_url = os.getenv('TIMESERIES_URL')
        self.api_key = self.authenticate()
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.datasource = DatasourceAPI(self.timeseries_url, self.headers)
        self.regression = RegressionAPI(self.timeseries_url, self.headers)
        self.classification = ClassificationAPI(self.timeseries_url, self.headers)
        self.clustering = ClusteringAPI(self.timeseries_url, self.headers)
        self.workflows = WorkflowsAPI(self.account_services, self.headers)

    def authenticate(self):
        response = requests.post(f"{self.account_services}/auth/token",                         
                                 json={"auth_key": self.auth_key, 
                                       "auth_secret": self.auth_secret})
        response.raise_for_status()
        return response.json()['accessToken']