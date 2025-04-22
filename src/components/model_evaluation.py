import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from sklearn.metrics import r2_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataIngestionArtifact
from src.logger import logging
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator

@dataclass
class EvaluatedModelResponse:
    is_model_accepted: bool
    changed_score: float
    trained_model_score: float
    best_model_score: float

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
        except MyException as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        try:
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_bucket_key
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _drop_id_column(self, df):
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered _create_dummy_columns in DataTransformation")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _remove_outliers_iqr(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logging.info("Removing outliers using IQR method")
        result_df = df.copy()
        total_rows_start = result_df.shape[0]

        for col in columns:
            if col == TARGET_COLUMN or col == 'Year':
                logging.info(f"Skipping outlier removal for column '{col}'")
                continue

            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask = (result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)
            rows_before = result_df.shape[0]
            result_df = result_df[mask]

            outliers_removed = rows_before - result_df.shape[0]
            logging.info(f"Removed {outliers_removed} outliers from column '{col}'")

        total_removed = total_rows_start - result_df.shape[0]
        logging.info(
            f"Total rows removed as outliers: {total_removed} ({(total_removed / total_rows_start * 100):.2f}% of data)")

        return result_df

    def evaluate_model(self) -> EvaluatedModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop([TARGET_COLUMN], axis=1), test_df[TARGET_COLUMN]

            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._remove_outliers_iqr(x, columns=x.select_dtypes(include=['int64', 'float64']).columns.tolist())
            y = y.loc[x.index]  # align y with filtered x

            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"Trained model loaded from {self.model_trainer_artifact.trained_model_file_path}")
            trained_model_score = self.model_trainer_artifact.metric_artifact.r2_score
            logging.info(f"Trained model score: {trained_model_score}")

            best_model_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info("Computing R2 score for production model")
                y_hat_best_model = best_model.predict(x)
                best_model_score = r2_score(y, y_hat_best_model)
                logging.info(f"Production model score: {best_model_score}, New R2 score: {trained_model_score}")

            tmp_best_model_score = 0 if best_model_score is None else best_model_score
            result = EvaluatedModelResponse(trained_model_score=trained_model_score,
                                            best_model_score=best_model_score,
                                            is_model_accepted=trained_model_score > tmp_best_model_score,
                                            changed_score=trained_model_score - tmp_best_model_score)
            logging.info("Evaluated model response {result}".format(result=result))
            return result
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initiating model evaluation")
            evaluated_model_response = self.evaluate_model()
            s3_model_file_path = self.model_evaluation_config.s3_bucket_key

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluated_model_response.is_model_accepted,
                changed_score=evaluated_model_response.changed_score,
                s3_model_file_path=s3_model_file_path,
                trained_model_file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Model evaluation result {result}".format(result=model_evaluation_artifact))
            return model_evaluation_artifact
        except MyException as e:
            raise MyException(e, sys)
