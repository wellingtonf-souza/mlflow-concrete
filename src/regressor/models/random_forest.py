from src.regressor.models.interface import Model
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
from src.logs import logger

class RandomForestRegressor(Model):
    def validate(self, algorithm: str)->bool:
        if algorithm == 'RandomForestRegressor': 
            return True
        return False

    def fit(self, grid_search: bool, x_train: pd.DataFrame, y_train: pd.Series)->RegressorMixin:
        if grid_search:
            logger.info("iniciando o fit do RandomForestRegressor com GridSearchCV")
            parameters = { 
                'n_estimators': [25, 50, 100, 200],
                'min_samples_leaf': [1, 2, 4, 6, 8]
            }
            grid = GridSearchCV(estimator = SkRandomForestRegressor(), param_grid = parameters, scoring = "explained_variance", cv=5)
            model = grid.fit(x_train, y_train)
        else:
            logger.info("iniciando o fit do RandomForestRegressor")
            model = SkRandomForestRegressor().fit(x_train, y_train)
        return model