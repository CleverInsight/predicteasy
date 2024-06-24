import os
from dotenv import load_dotenv
import requests
from .endpoints.regression import RegressionAPI
from .endpoints.datasource import DatasourceAPI
from .endpoints.classification import ClassificationAPI
from .endpoints.clustering import ClusteringAPI
from .endpoints.workflows import WorkflowsAPI
from predicteasy.exceptions import AuthenticationError

load_dotenv()

class PredictEasyClient:
    def __init__(self, auth_key, auth_secret):
        self.auth_key = auth_key
        self.auth_secret = auth_secret
        self.account_services = os.getenv('ACCOUNT_SERVICES')
        self.worker_url = os.getenv('WORKER_SERVICES')
        self.api_key = self.authenticate()
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.apis = {
            'datasource': DatasourceAPI(self.worker_url, self.headers),
            'regression': RegressionAPI(self.worker_url, self.headers),
            'classification': ClassificationAPI(self.worker_url, self.headers),
            'clustering': ClusteringAPI(self.worker_url, self.headers),
            'workflows': WorkflowsAPI(self.account_services, self.headers)
        }

    def authenticate(self):
        try:
            response = requests.post(f"{self.account_services}/auth/token",
                                     json={"auth_key": self.auth_key, "auth_secret": self.auth_secret})
            response.raise_for_status()
            return response.json()['accessToken']
        except requests.exceptions.HTTPError as e:
            raise AuthenticationError("Authentication failed. Please check your credentials.") from e 
            # Todo: to handle other HTTP errors or re-raise them

    def __getattr__(self, name):
        for api in self.apis.values():
            if hasattr(api, name):
                return getattr(api, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
