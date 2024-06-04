import requests

class WorkflowsAPI:
    def __init__(self, account_services, headers):
        self.base_url = f"{account_services}/workflows"
        self.headers = headers

    def get_workflows(self, page=1, limit=10):
        url = f"{self.base_url}/list?page={page}&limit={limit}"
        response = requests.get(url, headers=self.headers)
        return self.handle_response(response)

    def handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
