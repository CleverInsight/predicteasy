import pandas as pd 
import numpy as np
from predicteasy.core import metrics
from predicteasy.core.data import TableFrame
from predicteasy.core.supervised import Classifier


def test_dataframe():
    """Test dataframe converstion to tableframe

    Returns:
        [type]: [description]
    """
    data = pd.read_csv('data/train.csv')
    return TableFrame(data)

def test_dataframe_shape():
    return test_dataframe().shape

def run_model_scoring():
    df = TableFrame(pd.read_csv('data/train.csv'))
    X = df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(0)
    y = df['Survived']
    clf = Classifier(X, y)
    return clf.scoring(multiple=False)

print(run_model_scoring())