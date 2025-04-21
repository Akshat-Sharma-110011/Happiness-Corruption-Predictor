import os
import sys
import json
import numpy as np
import pandas as pd

from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH
from src.exception import MyException
from src.utils.main_utils import read_yaml_file
from src.logger import logging

class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating number of columns....")
            status = len(df.columns) == len(self._schema_config['columns'])
            logging.info(f"Is the Number of columns Present?: {status}")
            return status
        except Exception as e:
            raise MyException(e, sys)

    def is_column_exists(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating columns....")
            df_columns = df.columns
            missing_num_cols = []
            missing_cat_cols = []

            for col in self._schema_config['numerical_columns']:
                if col not in df_columns:
                    missing_num_cols.append(col)

            if len(missing_num_cols) > 0:
                logging.info(f"The missing numerical columns: {(missing_num_cols)}")

            for col in self._schema_config['categorical_columns']:
                if col not in df_columns:
                    missing_cat_cols.append(col)

            if len(missing_cat_cols) > 0:
                logging.info(f"The missing categorical columns: {(missing_cat_cols)}")

            return False if len(missing_cat_cols)>0 or len(missing_num_cols)>0 else True
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info("Reading data...")
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating data validation...")
            validation_error_msg = ""
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path), DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(train_df)
            if not status:
                validation_error_msg += f"Columns are missing in the training dataframe"
            else:
                logging.info(f"All required columns present in training dataframe: {status}")

            status = self.is_column_exists(train_df)
            if not status:
                validation_error_msg += f"Columns are missing in the training dataframe"
            else:
                logging.info(f"All required columns present in training dataframe: {status}")

            status = self.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg += f"Columns are missing in the test dataframe"
            else:
                logging.info(f"All required columns present in test dataframe: {status}")

            status = self.is_column_exists(test_df)
            if not status:
                validation_error_msg += f"Columns are missing in the test dataframe"
            else:
                logging.info(f"All required columns present in test dataframe: {status}")

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status= validation_status,
                message= validation_error_msg,
                validation_report_file_path= self.data_validation_config.data_validation_report_file_path
            )
            report_dir = os.path.dirname(self.data_validation_config.data_validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {"validation_status": validation_status, "message": validation_error_msg}

            with open(self.data_validation_config.data_validation_report_file_path, "w") as f:
                json.dump(validation_report, f, indent=4)

            logging.info(f"Data validation report saved to {self.data_validation_config.data_validation_report_file_path}")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e