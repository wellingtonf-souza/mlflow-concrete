import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import typer
import mlflow
import mlflow.sklearn
from src.logs import logger
from src.helpers import get_data
from typing import Tuple

def get_metrics(y_true: np.array, y_pred: np.array)->Tuple[float,float]:
    evs = explained_variance_score(y_true ,y_pred)
    mae = mean_absolute_error(y_true ,y_pred)
    return evs, mae

def train(
    algorithm: str = typer.Argument(
        "RandomForestRegressor", 
        help="Algoritmo a ser treinado com as seguintes opções: RandomForestRegressor, Ridge, KNeighborsRegressor e AdaBoostRegressor"
    ), 
    grid_search: bool = typer.Argument(
        False,
        help="Especifica se irá realizar a busca pelos melhores parâmetros ou utilizar os valores default de cada algoritmo")
    ):
    warnings.filterwarnings("ignore")
    if algorithm not in ['RandomForestRegressor', 'Ridge', 'KNeighborsRegressor', 'AdaBoostRegressor']:
        message = typer.style("algorithm incorreto \n", fg = typer.colors.WHITE, bg = typer.colors.RED)
        message = message + "É necessário que seja uma das seguintes opções: RandomForestRegressor, Ridge, KNeighborsRegressor ou AdaBoostRegressor"
        typer.echo(message)
    breakpoint()
    data = get_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns='compressive_strength'), 
        data['compressive_strength'], 
        test_size=0.3, 
        random_state=5463
    )

    model = Ridge().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    evs, mae = get_metrics(y_test, y_pred)
        
    mlflow.start_run()
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"evs": evs, "mae": mae})
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    typer.run(train)