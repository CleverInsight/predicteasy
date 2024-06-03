# predicteasy/schemas.py
from pydantic import BaseModel
from typing import List

class ClassificationRequest(BaseModel):
    datasource_id: str
    title: str
    test_size: float
    cross_val: int
    x: List[str]
    y: str

class RegressionRequest(BaseModel):
    datasource_id: str
    title: str
    test_size: float
    cross_val: int
    x: List[str]
    y: str

class ClusteringRequest(BaseModel):
    datasource_id: str
    title: str
    exclude: List[str]
    n_clusters: int

class CreateDatasourceRequest(BaseModel):
    title: str
    description: str
    horizontal: List[str]
    vertical: str
    file_path: str
