import time
import pandas as pd 
import numpy as np
import sklearn
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier, CatBoostRegressor




class Classifier:
    """
    A supervised learning classifier which can help us to 
    evaluate given dataset.
    """
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
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.X.get_numerical()),
                ('cat', categorical_transformer, self.X.get_categorical())   
            ],
            remainder='passthrough', verbose=True
        )
        # evaluate on multiple models
        for name, model  in models:
            start_time = time.time()
            etree = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
            etree.fit(self.X, self.y)
            dump(etree, name+'.joblib')
            end_time = time.time()
            elapsed_time = (end_time - start_time) 

            numerical_columns = self.X.get_numerical()
            onehot_columns = etree.named_steps['preprocessor']\
                .named_transformers_['cat'].named_steps['onehot']\
                .get_feature_names(input_features=self.X.get_categorical())

            #you can get the values transformed with your pipeline
            X_values = preprocessor.fit_transform(self.X)

            all_columns = list(numerical_columns) + list(onehot_columns)

            df_from_etree_pipeline = pd.DataFrame(X_values, columns=all_columns)

            # feature_importance = pd.Series(
            #     data=etree.named_steps['classifier'].feature_importances_, \
            #     index = np.array(all_columns))

            feature_importance = pd.DataFrame({
                    'Feature': np.array(all_columns),
                    'Score': etree.named_steps['classifier'].feature_importances_
            }).sort_values(by=['Score'], ascending=False)

            self.time_taken[name] = elapsed_time
            self.result[name] = feature_importance
            self.cross_score[name] = sklearn.model_selection.cross_val_score(
                etree, self.X, self.y, cv=4).mean()
        
        if (rtype=='dataframe'):
            return [{"elapsed": self.time_taken[name],
            "name": name,
            "cross_val": self.cross_score[name], 
            'model': result} for name, result in self.result.items()]
        elif (rtype=='dict'):
            return [{"elapsed": self.time_taken[name], 
            "name": name,
            "cross_val": self.cross_score[name], 
            'model': result.to_dict(orient='records')} for name, result in self.result]
        if (rtype=='json'):
            return [{"elapsed": self.time_taken[name], 
            "name": name,
            "cross_val": self.cross_score[name], 
            'model': result.to_json(orient='records')} for name, result in self.result.items()]



class Regressor:
    """
    A Supervised Regressor is a meta classifier which is used to 
    evalute the given the data.

    """

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
                ('extra_tree', sklearn.ensemble.ExtraTreesRegressor()),
                ('dtree', sklearn.tree.DecisionTreeRegressor()),
                ('rfc', sklearn.ensemble.RandomForestRegressor()),
                ('catboost', CatBoostRegressor())
            ]
        else:
            models = [
                ('extra_tree', sklearn.ensemble.ExtraTreesRegressor())
            ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(categories='auto', sparse=False, 
            handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.X.get_numerical()),
                ('cat', categorical_transformer, self.X.get_categorical())   
            ],
            remainder='passthrough', verbose=True
        )
        # evaluate on multiple models
        for name, model  in models:
            start_time = time.time()
            etree = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
            etree.fit(self.X, self.y)
            dump(etree, name+'.joblib')
            end_time = time.time()
            elapsed_time = (end_time - start_time) 

            numerical_columns = self.X.get_numerical()
            onehot_columns = etree.named_steps['preprocessor']\
                .named_transformers_['cat'].named_steps['onehot']\
                .get_feature_names(input_features=self.X.get_categorical())

            #you can get the values transformed with your pipeline
            X_values = preprocessor.fit_transform(self.X)

            all_columns = list(numerical_columns) + list(onehot_columns)

            df_from_etree_pipeline = pd.DataFrame(X_values, columns=all_columns)

            feature_importance = pd.DataFrame({
                    'Feature': np.array(all_columns),
                    'Score': etree.named_steps['classifier'].feature_importances_
            }).sort_values(by=['Score'], ascending=False)

            self.time_taken[name] = elapsed_time
            self.result[name] = feature_importance
            self.cross_score[name] = sklearn.model_selection.cross_val_score(
                etree, self.X, self.y, cv=4).mean()
        
        if (rtype=='dataframe'):
            return [{"elapsed": self.time_taken[name],
            "name": name,
            "cross_val": self.cross_score[name], 
            'model': result} for name, result in self.result.items()]
        elif (rtype=='dict'):
            return [{"elapsed": self.time_taken[name], 
            "name": name,
            "cross_val": self.cross_score[name], 
            'model': result.to_dict(orient='records')} for name, result in self.result]
        if (rtype=='json'):
            return [{"elapsed": self.time_taken[name], 
            "name": name,
            "cross_val": self.cross_score[name], 
            'model': result.to_json(orient='records')} for name, result in self.result.items()]
