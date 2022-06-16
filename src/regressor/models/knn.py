from src.regressor.models.interface import Model
from sklearn.neighbors import KNeighborsRegressor as SkKNeighborsRegressor
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
from src.logs import logger

class KNeighborsRegressor(Model):
    def validate(self, algorithm: str)->bool:
        if algorithm == 'KNeighborsRegressor': 
            return True
        return False

    def fit(self, grid_search: bool, x_train: pd.DataFrame, y_train: pd.Series)->RegressorMixin:
        if grid_search:
            logger.info("iniciando o fit do KNeighborsRegressor com GridSearchCV")
            parameters = {'n_neighbors':[3, 5, 7, 9]}
            grid = GridSearchCV(estimator = SkKNeighborsRegressor(), param_grid = parameters, scoring = "explained_variance", cv=5)
            model = grid.fit(x_train, y_train)
        else:
            logger.info("iniciando o fit do KNeighborsRegressor")
            model = SkKNeighborsRegressor().fit(x_train, y_train)
        return model