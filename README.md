
Formas de executar os experimentos:

~~~bash
mlflow run --experiment-name concrete . -P algorithm=Ridge -P grid_search=n
~~~

Para visualizar os resultados
~~~bash
mlflow ui
~~~