"""
TABLEFRAME.py

Data wrangling module

written by Bastin Robins and Dr.Vandana Bhagat
"""

import json
import time
import joblib
import string
import random
import logging
import tempfile
import numpy as np
import pandas as pd
from urllib import parse
from datetime import datetime
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
from sklearn import discriminant_analysis
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.model_selection import cross_val_score
# import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def rmsl_error(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


class TableFrame(pd.DataFrame):

    _metadata = ["_encoders", "_logs"]
    def __init__(self, *args, **kw):
        super(TableFrame, self).__init__(*args, **kw)
        self._encoders = []
        self._logs = []
    
    @property
    def _constructor(self):
        return TableFrame

    @classmethod
    def _internal_ctor(cls, *args, **kwargs):
        kwargs["_encoders"]  = []
        kwargs["_logs"] = [] 
        return cls(*args, **kwargs)

    def categorical(self):
        return self.select_dtypes(exclude=[np.number])

    def numerical(self):
        return self.select_dtypes(include=[np.number])

    def get_categorical(self):
        return self.categorical().columns

    def get_numerical(self):
        return self.numerical().columns


    def log_changes(self, key, value):
        self._logs.append(
            {
                "key": key,
                "value": value,
                "created": datetime.now()
            }
        )

    def log_transform(self, column):
        x = self[column].values
        self[column] = np.log(x, where=0<x, out=np.nan*x)
        self.log_changes(column, 'log')
        return self[column]

    def log2_transform(self, column):
        x = self[column].values
        self[column] = np.log2(x, where=0<x, out=np.nan*x)
        self.log_changes(column, 'log2')
        return self[column]

    def rename_col(self, mappers):
        self.rename(columns=mappers, inplace=True)
        for key, value in mappers.items():
            self.log_changes(key, 'rename')
        return self

    def scale(self, column):
        self[column] = preprocessing.scale(self[column])
        self.log_changes(column, 'scale')
        return self[column]

    def label_encode(self, column):
        le = preprocessing.LabelEncoder()
        self[column] = le.fit_transform(self[column])
        self.log_changes(column, 'label_encode')
        self._encoders.append({column : le.classes_ })
    
    def impute(self, column):
        if self[column] == 'object':
            self[column] = self[column].fillna(self[column].mode()[0])
            self.log_changes(column, 'impute')
        else:
            self[column] = self[column].fillna(self[column].mean())
        return self[column]


    def filter(self, column, operation, value):
        self.log_changes(column, 'filter')
        if operation == '==':
            return self[self[column]==value]
        elif operation == '>=':
            return self[self[column]>=value]
        elif operation == '<=':
            return self[self[column]<=value]
        elif operation == '>':
            return self[self[column]>value]
        elif operation == '<':
            return self[self[column]<value]
        elif operation == 'contain':
            return self[self[column].str.contains(value)]
        elif operation == 'isin':
            return self[self[column].isin(value)]

    
    def query_filter(self, query):
        """Query Filter takes query_string as parameter
        and filter the given dataframe using given query filter

        Args:
            query ([string]): [query string from url]

        Returns:
            [DataFrame]: [Filtered Subset DataFrame]
        """
        query_dict = parse.parse_qs(query)
        for key, values in query_dict.items():
            data = self[self[key].astype(str).isin(values)]
        return data


    def find_replace(self, column, origin, destination):
        """Replace the existing column with the given value

        Args:
            column ([type]): [description]
            origin ([type]): [description]
            destination ([type]): [description]

        Returns:
            [type]: [description]
        """
        temp =  self[column].copy()
        temp.replace(to_replace=origin, value=destination, inplace=True)
        self[column] = temp
        return self[column]


    
    async def factor_support(self, col, target, websocket):
        df = self[[col, target]]
        response = {}
        counter = 40

        if len(df[target].unique()) >= 10:
            total_bins = pd.cut(df[target], 4).unique()
            await websocket.send_text(json.dumps({"type": "message", "data": "Calculating ....", "percent": counter }, cls=NpEncoder))
            for index, target_bin in enumerate(total_bins):
                counter = counter + index
                bin_name = str(target_bin.left)+'-'+str(target_bin.right)
                small_df = df[(df[target] >= target_bin.left) & (df[target] <= target_bin.right)]
                response[bin_name] = [{"x": k, "y": v} for k, v in small_df[col].value_counts().to_dict().items()]
                msg = 'Detecting relationship between ' + str(col) + ' and ' + bin_name
                await websocket.send_text(json.dumps({"type": "message", "data": msg, "percent": counter }, cls=NpEncoder))
        else:
            await websocket.send_text(json.dumps({"type": "message", "data": "Calculating ....", "percent": counter }, cls=NpEncoder))
            for index, target_bin in enumerate(df[target].unique()):
                counter = counter + index
                small_df = df[df[target]==target_bin]
                response[str(target_bin)] = [{"x": k, "y": v} for k, v in small_df[col].value_counts().to_dict().items()]
                msg = 'Categorical relationship between ' + str(col)
                await websocket.send_text(json.dumps({"type": "message", "data": msg, "percent": counter }, cls=NpEncoder))

        return response

    # Detect multicollinearity
    async def calculate_vif(self, X):
        vif = pd.DataFrame()
        vif["variables"] = X.columns
        vif["VIF"] = [variance_inflation_factor(np.array(X.values, dtype=float), i) for i in range(X.shape[1])]
        return vif

    
    async def feature_scoring(self, target, model_type, websocket, exclude_list):
        """Feature Scoring From Internal Dataframe call

        Args:
            target ([type]): [description]
            model_type ([type]): [description]
            websocket ([type]): [description]
            exclude_list (list, optional): [description]. Defaults to [].

        Returns:
            [type]: [description]
        """
        exclude_list.append(target)
        await websocket.send_text(json.dumps({"type": "message", "data": 'Selecting Target', "percent": 10 }, cls=NpEncoder)) 

        for col in self.columns:
            await websocket.send_text(json.dumps({"type": "message", "data": 'Detecting dates..', "percent": 15 }, cls=NpEncoder)) 
            if self[col].dtype == 'object':
                try:
                    self[col] = pd.to_datetime(self[col])
                    await websocket.send_text(json.dumps({"type": "message", "data": 'Convert objects to dates..', "percent": 20 }, cls=NpEncoder)) 
                except ValueError:
                    pass

        # Get the target value
        y = self[target].astype('int')

        # Exclude all columns which are given in list
        X = self[self.columns.difference(exclude_list)]
        features = X.columns      


        await websocket.send_text(json.dumps({"type": "message", "data": 'Selecting Features', "percent": 35  }, cls=NpEncoder))

        if model_type=='categorical':
            etree = ensemble.ExtraTreesClassifier()
            await websocket.send_text(json.dumps({"type": "message", "data": 'Classification model initiating', "percent": 60  })) 

        else:
            etree = ensemble.ExtraTreesRegressor()
            await websocket.send_text(json.dumps({"type": "message", "data": 'Regression model initiating', "percent": 60 })) 

        await websocket.send_text(json.dumps({"type": "message", "data": 'Training model started', "percent": 70 })) 
        etree.fit(X.values, y)
        await websocket.send_text(json.dumps({"type": "message", "data": 'Calculating model accuracy', "percent": 75 })) 
        await websocket.send_text(json.dumps({"type": "message", "data": 'Scanning for multicollinearity Features', "percent": 25  }, cls=NpEncoder))
        vif = await self.calculate_vif(X)

        await websocket.send_text(json.dumps({"type": "message", "data": 'Completed.', "percent": 25  }, cls=NpEncoder))
        

        count, mean, median, unique, colinear, response  = [],[],[],[],[],{}
        counter = 75
        for index, col in enumerate(X.columns):
            await websocket.send_text(json.dumps({"type": "message", "data": 'Completed '+ str(col), "percent": 25  }, cls=NpEncoder))
            counter = counter + index
            count.append(self[col].count())
            mean.append(self[col].mean())
            median.append(self[col].median())
            unique.append(len(self[col].unique()))
            if col in vif['variables'].tolist():
                colinear.append(vif[vif['variables']==col]['VIF'].values[0])
            else:
                colinear.append('No')
            response[col] = await self.factor_support(col, target, websocket)
            await websocket.send_text(json.dumps({"type": "message", "data": 'Calculating mean,median,count,unique for '+ str(col), "percent": counter })) 


        summary = {
            'feature': features,
            'type': ['numerical' if self[col].dtypes in ['int64', 'float64'] else 'discrete' for col in features],
            'score': etree.feature_importances_,
            'unique': unique,
            'mean': mean,
            'median': median,
            'count': count,
            'colinear': colinear
        }

        # important_features = pd.Series(data=etree.feature_importances_, index=X.columns)
        important_features = pd.DataFrame(summary)
        important_features.sort_values(by=['score'],ascending=False,inplace=True)
        
        await websocket.send_text(json.dumps({"type": "message", "data": 'Feature scoring completed', "percent": 90 })) 
        return important_features, response




    async def ensemble_score(self, target, model_type, websocket, app_id, user_id, exclude_list=[]):
        await websocket.send_text(json.dumps({"type": "message", "data": "Importing different models", "percent": 5 }))
        all_estimators = {
            'categorical': {
          
                "gradient_boost_c": {
                    "name": "Gradient Boosting Classifier",
                    "model": ensemble.GradientBoostingClassifier(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "rf_c": {
                    "name": "RandomForest Classifier",
                    "model": ensemble.RandomForestClassifier(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "dt_c": {
                    "name": "Decision Tree Classifier",
                    "model": tree.DecisionTreeClassifier(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "ada_c": {
                    "name": "Ada Boosting",
                    "model": ensemble.AdaBoostClassifier(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "lda_c": {
                    "name": "Linear Discriminant Analysis",
                    "model": discriminant_analysis.LinearDiscriminantAnalysis(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "ridge_c": {
                    "name": "Ridge Classifier",
                    "model": linear_model.RidgeClassifier(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "logistic_c": {
                    "name": "Logistic Regression",
                    "model": linear_model.LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "knn_c": {
                    "name": "K Neighbours Classifier",
                    "model": neighbors.KNeighborsClassifier(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                },
                "qda_c": {
                    "name": "Quadratic Discriminant Analysis",
                    "model": discriminant_analysis.QuadraticDiscriminantAnalysis(),
                    "cost_fn": ['Accuracy', 'AUC', 'Recall', 'Precision', 'Cohen Kappa']
                }

            },
            'numerical': {
                "linear_r": {
                    "name": "Linear Regression",
                    "model": linear_model.LinearRegression(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "lasso_r": {
                    "name": "Lasso Regression",
                    "model": linear_model.Lasso(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "ridge_r": {
                    "name": "Ridge Regression",
                    "model": linear_model.Ridge(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "enet_r": {
                    "name": "Elastic Net",
                    "model": linear_model.ElasticNet(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "least_angle_r": {
                    "name": "Least Angle Regression",
                    "model": linear_model.Lars(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "lasso_least_angle_r": {
                    "name": "Lasso Least Angle Regression",
                    "model": linear_model.LassoLars(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
            
                "gradient_boost_r": {
                    "name": "Gradient Boosting Regression",
                    "model": ensemble.GradientBoostingRegressor(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "rf_r": {
                    "name": "RandomForest Regression",
                    "model": ensemble.RandomForestRegressor(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "dt_r": {
                    "name": "Decision Tree Regression",
                    "model": tree.DecisionTreeRegressor(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "ada_r": {
                    "name": "Ada Boosting Regression",
                    "model": ensemble.AdaBoostRegressor(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
        
                "svm_linear_r": {
                    "name": "SVM - Linear Kernel",
                    "model": svm.SVR(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "knn_r": {
                    "name": "K Neighbours Classifier",
                    "model": neighbors.KNeighborsRegressor(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                },
                "naive_r": {
                    "name": "Naive Bayes",
                    "model": linear_model.BayesianRidge(),
                    "cost_fn": ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
                }
            }
        }

        await websocket.send_text(json.dumps({"type": "message", "data": model_type, "percent": 6 }))
        
        exclude_list.append(target)
        await websocket.send_text(json.dumps({"type": "message", "data": 'Selecting Target', "percent": 7 }, cls=NpEncoder)) 

        for col in self.columns:
            await websocket.send_text(json.dumps({"type": "message", "data": 'Detecting dates..', "percent": 8 }, cls=NpEncoder)) 
            if self[col].dtype == 'object':
                try:
                    self[col] = pd.to_datetime(self[col])
                    await websocket.send_text(json.dumps({"type": "message", "data": 'Convert objects to dates..', "percent": 9 }, cls=NpEncoder)) 
                except ValueError:
                    pass


        # Exclude all columns which are given in list
        X = self[self.columns.difference(exclude_list)]
        features = X.columns        

        await websocket.send_text(json.dumps({"type": "message", "data": 'Selecting Features', "percent": 10  }, cls=NpEncoder))

        if model_type=='categorical':
            y = self[target].astype('int')
            estimators = all_estimators[model_type]
            await websocket.send_text(json.dumps({"type": "message", "data": 'Classification models initiating', "percent": 11  })) 

        else:
            y = self[target].astype('float')
            estimators = all_estimators[model_type]
            await websocket.send_text(json.dumps({"type": "message", "data": 'Regression models initiating', "percent": 11 })) 

        counter = 10
        # skf = model_selection.StratifiedKFold(n_splits=2, random_state=None)
        await websocket.send_text(json.dumps({"type": "message", "data": 'Cross validation initiated', "percent": counter })) 

        model_summary = []
        for index, model in enumerate(estimators.keys()):

            logger.info(estimators[model]['name'] + ' estimator initiating')
            score_auc = np.empty((0,0))
            score_recall = np.empty((0,0))
            score_acc = np.empty((0,0))
            score_precision = np.empty((0,0))
            score_kappa = np.empty((0,0))
            score_f1 = np.empty((0,0))
            score_mcc = np.empty((0,0))
            score_mae = np.empty((0,0))
            score_mse = np.empty((0,0))
            score_rmse = np.empty((0,0))
            score_r2 = np.empty((0,0))
            score_rmsle = np.empty((0,0))
            score_mape = np.empty((0,0))
     

            start = time.time()
            # # for train_index, test_index in skf.split(X, y):
            counter = counter + index
            # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            # y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=.3) 

            await websocket.send_text(json.dumps({"type": "message", "data":  estimators[model]['name'] + ' estimator initiating', "percent": counter })) 
            clf = estimators[model]['model']
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            logger.info(estimators[model]['name'] + ' estimator completed')

            if model_type == 'categorical':

                # score_auc = np.append(score_auc, metrics.roc_auc_score(y_pred, y_test, multi_class="ovo"))
                score_acc = np.append(score_acc, metrics.accuracy_score(y_pred, y_test))
                score_recall = np.append(score_recall, metrics.recall_score(y_pred, y_test, average=None))
                score_precision = np.append(score_precision, metrics.precision_score(y_pred, y_test, average=None))
                score_kappa = np.append(score_kappa, metrics.cohen_kappa_score(y_pred, y_test))
                score_mcc = np.append(score_mcc, metrics.matthews_corrcoef(y_pred, y_test))
                score_f1 = np.append(score_f1, metrics.f1_score(y_pred, y_test, average='micro'))

            if model_type == 'numerical':
                score_acc = np.append(score_acc, metrics.explained_variance_score(y_test, y_pred))
                score_mae = np.append(score_mae, metrics.mean_absolute_error(y_test, y_pred))
                score_mse = np.append(score_mse, metrics.mean_squared_error(y_test, y_pred))
                score_rmse = np.append(score_rmse, metrics.mean_squared_error(y_test, y_pred, squared=False))
                score_r2 = np.append(score_r2, metrics.r2_score(y_test, y_pred))
                score_rmsle = np.append(score_rmsle, np.sqrt(np.mean(
                    np.square(np.log1p(y_test.ravel() - y_test.ravel().min() + 1) - np.log1p(y_pred.ravel() - y_pred.ravel().min() + 1)))))
                # print("RMSE", y_test)
                # print("RMSE", y_test.shape)
                # print("RMSE2", y_pred.shape)
                # print("Predicted", y_pred)
            
            
            await websocket.send_text(json.dumps({"type": "message", "data": estimators[model]['name'] + ' accuracy', "percent": counter })) 
            end = time.time()

            summary = {}
            for col in features:
                summary[col] = {
                    "min": X[col].min(),
                    "max": X[col].max(),
                    "default": X[col].iloc[0],
                    "dtype": str(X[col].dtype)
                }


            if model_type == 'categorical':
                temp = {
                    "name": estimators[model]['name'],
                    "type": model_type,
                    "scores": score_acc,
                    "accuracy": score_acc.mean(),
                    "time_elasped": (end - start),
                    # "auc": score_auc.mean(),
                    'f1': score_f1.mean(),
                    "recall": score_recall.mean(),
                    "precision": score_precision.mean(),
                    "kappa": score_kappa.mean(),
                    "features": summary,
                    "target": target
                }
                
                # app_id, model_id, name, score)
                await websocket.send_text(json.dumps({"type": "message", "data": estimators[model]['name'] + ' saving into cloud', "percent": counter }, cls=NpEncoder)) 
                model_id, model_path = await self.save_model_to_s3(clf, app_id, user_id, estimators[model]['name'], score_acc.mean(), json.dumps(temp, cls=NpEncoder))
                temp['model_id'] = model_id
                model_summary.append(temp)

            else:
                temp = {
                    "name": estimators[model]['name'],
                    "type": model_type,
                    "scores": score_acc,
                    "accuracy": score_acc.mean(),
                    "time_elasped": (end - start),
                    "mae": score_mae.mean(),
                    "mse": score_mse.mean(),
                    "rmse": score_rmse.mean(),
                    "rmsle": score_rmsle.mean(),
                    "r2": score_r2.mean(),
                    "features": summary,
                    "target": target
                }

                await websocket.send_text(json.dumps({"type": "message", "data": estimators[model]['name'] + ' saving into cloud', "percent": counter }, cls=NpEncoder)) 
                model_id, model_path = await self.save_model_to_s3(clf, app_id, user_id, estimators[model]['name'], score_acc.mean(), json.dumps(temp, cls=NpEncoder))
                temp['model_id'] = model_id
                model_summary.append(temp)
        return model_summary
