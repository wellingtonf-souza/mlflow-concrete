
# MLflow

O [MLflow](https://mlflow.org/) é uma plataforma de código aberto para gerenciar o ciclo de vida de ML, incluindo experimentação, reprodutibilidade, implantação e um registro de modelo central. 

Este repositório foi desenvolvido como parte dos requisitos do modulo de MLOps do MBA em [Machine Learning in Production](https://iti.ufscar.mba/) da UFSCar. 

Optei pelo desenvolvimento de alguns experimentos para o problema de previsão de resistência do concreto à compressão cuja base de dados encontra-se no [UCI](https://archive-beta.ics.uci.edu/ml/datasets/concrete+compressive+strength). Este conjunto de dados apresenta as seguintes variáveis:

* Cimento (kg por m3)
* Escória de Alto Forno (kg por m3)
* Cinza volante (kg por m3)
* Água (kg por m3)
* Superplastificante (kg por m3)
* Agregado grosso (kg por m3)
* Agregado Fino (kg por m3)
* Idade em dias (1~365)
* Resistência à compressão do concreto

É possível executar os modelos RandomForestRegressor, Ridge, KNeighborsRegressor e AdaBoostRegressor, com ou sem o GridSearch. Para executar o projeto é necessário possuir instalados o conda (ou miniconda) e o mlflow.

Para executar, por exemplo, o Ridge sem busca de hiperparâmetros basta rodar o seguinta comando:

~~~bash
mlflow run --experiment-name concrete . -P algorithm=Ridge -P grid_search=n
~~~

Para visualizar os resultados basta subir a plataforma web com o comando abaixo e acessar http://localhost:5000/.
~~~bash
mlflow ui
~~~