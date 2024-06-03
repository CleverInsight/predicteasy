import requests
from pydantic import ValidationError
from ..schemas import ClusteringRequest
from IPython.display import IFrame 

class ClusteringAPI:
    def __init__(self, timeseries_url, headers):
        self.base_url = f"{timeseries_url}/models/clustering/"
        self.headers = headers

    def cluster(self, datasource_id: str, title: str, exclude: list, n_clusters: int):
        try:
            request = ClusteringRequest(
                datasource_id=datasource_id,
                title=title,
                exclude=exclude,
                n_clusters=n_clusters
            )
        except ValidationError as e:
            return {"error": str(e)}

        data = {
            "mode": "datasource",
            "title": request.title,
            "datasource_id": request.datasource_id,
            "exclude": request.exclude,
            "n_clusters": request.n_clusters,
            "meta": {
                "source": "url",
                "service": "63ee177cad2ecfb063d83dfa"
            }
        }
        response = requests.post(self.base_url, json=data, headers=self.headers)
        return self.handle_response(response)

    def handle_response(self, response):
        if response.status_code == 200:
            response_json = response.json()
            workflow_id = response_json.get('workflow_id')
            iframe_url = f"https://sheets.predicteasy.com/reports/{workflow_id}/{workflow_id}"
            return IFrame(iframe_url, width="100%", height=600)
        else:
            response.raise_for_status()
