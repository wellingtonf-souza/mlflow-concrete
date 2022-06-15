import logging

class Logger:
    @staticmethod
    def get_config():
        logger = logging.getLogger()

        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler('src/logs/mlflow.log')
        stream_handler = logging.StreamHandler()

        stream_formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
        file_formatter = logging.Formatter("{'timestamp': '%(asctime)s','level': '%(levelname)s','message': '%(message)s','filename':'%(filename)s','funcName':'%(funcName)s','levelno':'%(levelno)s','lineno':'%(lineno)s','module':'%(module)s'}")

        file_handler.setFormatter(file_formatter)
        stream_handler.setFormatter(stream_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger