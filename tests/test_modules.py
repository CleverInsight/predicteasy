import pandas as pd 
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from predicteasy.core import metrics
from predicteasy.core.data import TableFrame
from predicteasy.core.supervised import Classifier


def run_model_scoring():
    df = TableFrame(pd.read_csv('data/train.csv'))
    X = df[['Sex','PassengerId', 'Pclass', 'Ticket', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']
    clf = Classifier(X, y)
    return clf.scoring(multiple=False)

models = run_model_scoring()
x = PrettyTable()
x.field_names = ["ID", "Model", "Score", "Elapsed"]
for (index, model) in enumerate(models):
    x.add_row([index+1, model['name'], model['elapsed'], model['cross_val']])
print(x)


def test_trained_model():
    df = TableFrame(pd.read_csv('data/train.csv'))
    X = df[['Sex','PassengerId', 'Pclass', 'Ticket', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare']].head(20)
    y = df['Survived'].head(20)
    clf = load('extra_tree.joblib')
    predicted = clf.predict(X)
    print(accuracy_score(predicted, y))
    print(confusion_matrix(predicted, y))


test_trained_model()
