from src.regressor.models.interface import Model
from sklearn.linear_model import Ridge as SkRidge
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
from src.logs import logger

class Ridge(Model):
    def validate(self, algorithm: str)->bool:
        if algorithm == 'Ridge': 
            return True
        return False

    def fit(self, grid_search: bool, x_train: pd.DataFrame, y_train: pd.Series)->RegressorMixin:
        if grid_search:
            logger.info("iniciando o fit do Ridge com GridSearchCV")
            parameters = {'alpha':[0.1, 1.0, 10.0]}
            grid = GridSearchCV(estimator = SkRidge(), param_grid = parameters, scoring = "explained_variance", cv=5)
            model = grid.fit(x_train, y_train)
        else:
            logger.info("iniciando o fit do Ridge")
            model =  SkRidge().fit(x_train, y_train)
        return model