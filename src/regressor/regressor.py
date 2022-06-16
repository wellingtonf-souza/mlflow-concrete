from src.regressor.models.interface import Model
import pandas as pd

class Regressor:

    def __init__(self, algorithm: str, grid_search: bool)->None:
        self.algorithm = algorithm 
        self.grid_search = grid_search

    def get(self, x_train: pd.DataFrame, y_train: pd.Series):
        models = Model.__subclasses__()
        for element in models:
            model = element()
            if model.validate(algorithm = self.algorithm):
                return model.fit(grid_search=self.grid_search, x_train = x_train, y_train = y_train)