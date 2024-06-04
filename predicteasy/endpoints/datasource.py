import requests
import pandas as pd
from pydantic import BaseModel
from ..schemas import DataSourceCreationModel
from typing import List, Optional
from datetime import datetime
import json

# List Datasource Single Response
class Datasource(BaseModel):
    _id: str 
    title: str
    description: Optional[str]
    meta: str
    tags: List[str] = []
    createdBy: str
    created_at: datetime
    updated_at: datetime

# Get Datasource Response
class DataSourceDetails:
    def __init__(self, data):
        self.content = data['content']
        self.datasource = data['datasource']

    def describe(self):
        return pd.DataFrame(self.content['description'])

    def sample(self):
        return pd.DataFrame(self.content['sample'])
    
    def details(self):
        return self.datasource

class DatasourceAPI:
    def __init__(self, worker_url, headers):
        self.base_url = f"{worker_url}/datasources/"
        self.headers = headers
    
    # list all datasources
    def listDatasource(self) -> List[Datasource]:
        response = requests.get(self.base_url, headers=self.headers)
        return self.handle_response(response)
    
    # get datasource by ID 
    def getDatasource(self, datasource_id: str) -> DataSourceDetails:
        url = f"{self.base_url}{datasource_id}"
        response = requests.get(url, headers=self.headers)
        return DataSourceDetails(self.handle_response(response))

    # Creation of Datasource
    def createDatasource(self, title: str, description: str, horizontal: List[str], vertical: str, file_path: str):
            url = f"{self.base_url}create/file"

        
            data = DataSourceCreationModel(
                title=title,
                description=description,
                horizontal=horizontal,
                vertical=vertical
            )

            data_dict = data.model_dump()
            # add CSV File
            with open(file_path, 'rb') as file:
                files = {
                    'file': (file_path, file, 'text/csv')
                }

                form_data = {
                    'title': data_dict['title'],
                    'description': data_dict['description'],
                    'horizontal': json.dumps(data_dict['horizontal']), 
                    'vertical': data_dict['vertical']
                }

                response = requests.post(url, data=form_data, files=files, headers=self.headers)                
                return self.handle_response(response)
            
    # TODO: API to be implemented
    def updateDatasource(self, datasource_id: str, data) -> Datasource:
        url = f"{self.base_url}/{datasource_id}"
        response = requests.put(url, json=data, headers=self.headers)
        return self.handle_response(response, model=Datasource)
    
    # Delete datasource by ID
    def deleteDatasource(self, datasource_id: str):
        url = f"{self.base_url}{datasource_id}"
        response = requests.delete(url, headers=self.headers)
        return self.handle_response(response)
    
    def handle_response(self, response, model: BaseModel = None):
        if response.status_code == 200:
            if model:
                return model.model_validate(response.json())
            return response.json()
        else:
            response.raise_for_status()