from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
import pandas as pd

class Model(ABC):

    @abstractmethod
    def validate(self, algorithm: str)->bool:
        raise NotImplementedError

    @abstractmethod
    def fit(self, grid_search: bool, x_train: pd.DataFrame, y_train: pd.Series)->RegressorMixin:
        raise NotImplementedError