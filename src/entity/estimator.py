import sys
import pandas as pd
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import MyException

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Predicting...")

            transformed_features = self.preprocessing_object.transform(X)
            predictions = self.trained_model_object.predict(transformed_features)

            return predictions
        except Exception as e:
            logging.error(e)
            raise MyException(e, sys)

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}"

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}"