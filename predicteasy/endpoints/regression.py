import requests
from pydantic import ValidationError
from ..schemas import RegressionRequest
from IPython.display import IFrame

class RegressionAPI:
    def __init__(self, timeseries_url, headers):
        self.base_url = f"{timeseries_url}/models/regression/"
        self.headers = headers

    def regression(self, datasource_id: str, title: str, test_size: float, cross_val: int, x: list, y: str):
        try:
            request = RegressionRequest(
                datasource_id=datasource_id,
                title=title,
                test_size=test_size,
                cross_val=cross_val,
                x=x,
                y=y
            )
        except ValidationError as e:
            return {"error": str(e)}

        data = {
            "mode": "datasource",
            "title": request.title,
            "datasource_id": request.datasource_id,
            "props": {
                "label": True,
                "cross_val": request.cross_val,
                "test_size": request.test_size,
                "x": request.x,
                "y": request.y
            },
            "meta": {
                "source": "url",
                "service": "63ee177cad2ecfb063d83df9"
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

