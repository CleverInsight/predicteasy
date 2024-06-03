import requests
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json

class Datasource(BaseModel):
    _id: str 
    title: str
    description: Optional[str]
    meta: str
    tags: List[str] = []
    createdBy: str
    created_at: datetime
    updated_at: datetime

class Datasources(BaseModel):
    datasources: List[Datasource]
    
class DatasourcePayload(BaseModel):
    title: str = Field(..., title="Title of the datasource")
    description: str = Field(..., title="Description of the datasource")
    horizontal: List[str] = Field(..., title="Horizontal category")
    vertical: str = Field(..., title="Vertical category")


class DatasourceAPI:
    def __init__(self, timeseries_url, headers):
        self.base_url = f"{timeseries_url}/datasources/"
        self.headers = headers
    
    def listDatasource(self) -> Datasources:
        response = requests.get(self.base_url, headers=self.headers)
        return self.handle_response(response, model=Datasources)
        
    
    def getDatasource(self, datasource_id: str) -> Datasource:
        url = f"{self.base_url}{datasource_id}"
        response = requests.get(url, headers=self.headers)
        return self.handle_response(response)

    def createDatasource(self, title: str, description: str, horizontal: List[str], vertical: str, file_path: str):
            url = f"{self.base_url}create/file"

            data = DatasourcePayload(
                title=title,
                description=description,
                horizontal=horizontal,
                vertical=vertical
            )

            data_dict = data.model_dump()

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

    def update(self, datasource_id: str, data) -> Datasource:
        url = f"{self.base_url}/{datasource_id}"
        response = requests.put(url, json=data, headers=self.headers)
        return self.handle_response(response, model=Datasource)

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
