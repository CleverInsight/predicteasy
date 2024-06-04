from pydantic import BaseModel, Field
from typing import List, Literal
from typing_extensions import Annotated

# Regression payload
class RegressionRequest(BaseModel):
    datasource_id: str
    title: str
    test_size: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    cross_val: Annotated[int, Field(strict=True, ge=2, le=10)]
    x: List[str]
    y: str

# Classification payload
class ClassificationRequest(BaseModel):
    datasource_id: str
    title: str
    test_size: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    cross_val: Annotated[int, Field(strict=True, ge=2, le=10)]
    x: List[str]
    y: str

# Cluster payload
class ClusteringRequest(BaseModel):
    datasource_id: str
    title: str
    exclude: List[str]
    n_clusters: int

# Datasource create payload
class DataSourceCreationModel(BaseModel):
    title: str = Field(..., title="Title of the datasource")
    description: str = Field(..., title="Description of the datasource")
    horizontal: List[Literal['Human Resources', 'Support', 'Quality Assurance', 'Sales & Marketing', 'Supply Chain', 'Logistics', 'Research & Development', 'CRM']] = Field(..., title="Horizontal category")
    vertical: Literal[
        'Cyber security', 'IT', 'Healthcare', 'Manufacturing', 'Banking & Finance', 'Telecom', 
        'Entertainment & Media', 'Fashion', 'Sports', 'Public Services', 'Hospitality', 
        'Retail & E-commerce', 'Real estate & construction', 'Education', 'Agriculture'
    ] = Field(..., title="Vertical category")