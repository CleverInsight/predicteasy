import time
import pandas as pd 
import numpy as np
import sklearn
from catboost import CatBoostClassifier



class Classifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.scores = []
        self.features = []
        self.result = {}
        self.cross_score = {}
        self.time_taken = {}

    def scoring(self, rtype='dataframe', multiple=False):
        if multiple==True:
            models = [
                ('extra_tree', sklearn.ensemble.ExtraTreesClassifier()),
                ('dtree', sklearn.tree.DecisionTreeClassifier()),
                ('rfc', sklearn.ensemble.RandomForestClassifier()),
                ('catboost', CatBoostClassifier())
            ]
        else:
            models = [
                ('extra_tree', sklearn.ensemble.ExtraTreesClassifier())
            ]
        for name, etree  in models:
            start_time = time.time()
            self.features = self.X.columns
            etree.fit(self.X, self.y)
            end_time = time.time()
            elapsed_time = end_time - start_time 
            self.scores = etree.feature_importances_
            self.time_taken[name] = elapsed_time
            self.result[name] = pd.DataFrame(
                        {'features': self.features, 
                        'coefficient': self.scores,
                        'score': self.scores*100
                }).sort_values(by=['score'], ascending=False)
            self.cross_score[name] = sklearn.model_selection.cross_val_score(
                etree, self.X, self.y, cv=4).mean()
            
        if (rtype=='dataframe'):
            return [{"elapsed": self.time_taken[name], "cross_val": self.cross_score[name], name: result} for name, result in self.result.items()]
        elif (rtype=='dict'):
            return [{"elapsed": self.time_taken[name], "cross_val": self.cross_score[name], name: result.to_dict(orient='records')} for name, result in self.result]
        if (rtype=='json'):
            return [{"elapsed": self.time_taken[name], "cross_val": self.cross_score[name], name: result.to_json(orient='records')} for name, result in self.result.items()]

