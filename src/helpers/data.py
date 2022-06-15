import pandas as pd
from src.logs import logger

def get_data()->pd.DataFrame:
    try:
        logger.info("iniciando o metodo get_data")
        data = pd.read_excel("src/data/raw/concrete_data.xls")
        data.columns = [
            'cement','slag','fly_ash','water','superplasticizer','coarse_aggr','fine_aggr','age','compressive_strength'
        ]
        return data
    except Exception as e:
        logger.error(f"iniciando o metodo get_data - erro: {e}")