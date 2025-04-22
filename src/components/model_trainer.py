import sys
from typing import Tuple

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, RegressionMetricsArtifact, DataTransformationArtifact
from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, object]:
        try:
            logging.info("Training model...")

            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            model = XGBRegressor(
                n_estimators=self.model_trainer_config._n_estimators,
                learning_rate=self.model_trainer_config._learning_rate,
                max_depth=self.model_trainer_config._max_depth,
                subsample=self.model_trainer_config._subsample,
                colsample_bytree=self.model_trainer_config._colsample_bytree
            )

            logging.info("Fitting model...")
            model.fit(x_train, y_train)
            logging.info("Testing model...")

            y_hat = model.predict(x_test)
            mae = mean_absolute_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mse = mean_squared_error(y_test, y_hat)
            rmse = np.sqrt(mse)

            logging.info(f"Model Evaluation Metrics:")
            logging.info(f"Mean Absolute Error (MAE): {mae}")
            logging.info(f"Mean Squared Error (MSE): {mse}")
            logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
            logging.info(f"R² Score: {r2}")

            metric_artifact = RegressionMetricsArtifact(
                mean_squared_error=mse,
                mean_absolute_error=mae,
                r2_score=r2,
                root_mean_squared_error=rmse
            )
            return model, metric_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train data loaded.")
            logging.info("Test data loaded.")

            trained_model, metric_artifact = self.get_model_object_and_report(train_arr, test_arr)
            logging.info("Model trained.")

            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_path)
            logging.info("Preprocessing object loaded.")

            if metric_artifact.r2_score < self.model_trainer_config.expected_accuracy:
                logging.info(f"Model R² Score ({metric_artifact.r2_score}) is less than expected ({self.model_trainer_config.expected_accuracy}). Rolling back changes.")
                raise Exception("Model accuracy isn't greater than the expected threshold")

            logging.info(f"Model R² Score passed threshold: {metric_artifact.r2_score}")

            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.model_trainer_trained_model_dir, my_model)
            logging.info(f"Model trained and saved to: {self.model_trainer_config.model_trainer_trained_model_dir}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.model_trainer_trained_model_dir,
                metric_artifact=metric_artifact
            )
            logging.info("Model training artifact created.")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e
