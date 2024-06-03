from pydantic import BaseModel, Field
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

class DataSourceCreationModel(BaseModel):
    title: str = Field(..., title="Title of the datasource")
    description: str = Field(..., title="Description of the datasource")
    horizontal: List[str] = Field(..., title="Horizontal category")
    vertical: str = Field(..., title="Vertical category")
