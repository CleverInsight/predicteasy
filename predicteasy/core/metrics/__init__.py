import pandas as pd 
import numpy as np 
from sklearn.ensemble import ExtraTreesClassifier


class Scorer:
    """
    A Scorer is a special classifier used to score
    for given X and y and generate statistical 
    significance coefficients. 

    ...
    
    Attributes:
    ----------
    X : tableframes
        X is the list of all features tableframe
    y : series or numpy series
        y is the series of the target variable
    
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def set_x(self, X):
        self.X = X
    
    def get_x(self):
        return self.X
    
    def set_y(self, y):
        self.y = y

    def get_y(self):
        return self.y
