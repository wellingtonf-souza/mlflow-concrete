from src.regressor.models.interface import Model
from sklearn.ensemble import AdaBoostRegressor as SkAdaBoostRegressor
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
from src.logs import logger

class AdaBoostRegressor(Model):
    def validate(self, algorithm: str)->bool:
        if algorithm == 'AdaBoostRegressor': 
            return True
        return False

    def fit(self, grid_search: bool, x_train: pd.DataFrame, y_train: pd.Series)->RegressorMixin:
        if grid_search:
            logger.info("iniciando o fit do AdaBoostRegressor com GridSearchCV")
            parameters = { 
                'n_estimators': [25, 50, 100],
                'base_estimator__min_samples_leaf': [1, 2, 4, 6],
                'base_estimator__max_depth': [3, 5, 7]
             }
            grid = GridSearchCV(estimator = SkAdaBoostRegressor(), param_grid = parameters, scoring = "explained_variance", cv=5)
            model = grid.fit(x_train, y_train)
        else:
            logger.info("iniciando o fit do AdaBoostRegressor")
            model = SkAdaBoostRegressor().fit(x_train, y_train)
        return model