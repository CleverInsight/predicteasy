import pandas as pd 
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from predicteasy.core import metrics
from Zauthentication.core.data import TableFrame
from Zauthentication.core.supervised import Classifier, Regressor


def run_model_scoring():
    df = TableFrame(pd.read_csv('data/train.csv'))
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    clf = Classifier(X, y)
    return clf.scoring(multiple=False)


def run_regressor_model():
    df = TableFrame(pd.read_csv('data/train.csv'))
    X = df[['Sex','PassengerId', 'Pclass', 'Ticket', 'Embarked', 'Survived', 'SibSp', 'Parch', 'Fare']]
    y = df['Age'].fillna(df['Age'].median())
    clf = Regressor(X, y)
    return clf.scoring(multiple=True)


def test_trained_clf_model():
    df = TableFrame(pd.read_csv('data/train.csv'))
    X = df[['Sex','PassengerId', 'Pclass', 'Ticket', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare']].head(20)
    y = df['Survived'].head(20)
    clf = load('extra_tree.joblib')
    predicted = clf.predict(X)
    print(accuracy_score(predicted, y))
    print(confusion_matrix(predicted, y))




    
    # clf = load('extra_tree.joblib')
    # predicted = clf.predict(X)
    # print(accuracy_score(predicted, y))
    # print(confusion_matrix(predicted, y))


# test_trained_clf_model()
models = run_model_scoring()
x = PrettyTable()
x.field_names = ["ID", "Model", "Score", "Elapsed"]
for (index, model) in enumerate(models):
    x.add_row([index+1, model['name'], model['elapsed'], model['cross_val']])
print(x)
