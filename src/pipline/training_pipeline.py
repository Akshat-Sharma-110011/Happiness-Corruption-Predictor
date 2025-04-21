import sys

from src.logger import logging
from src.exception import MyException

from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,)
from src.entity.artifact_entity import (DataIngestionArtifact,
                                        DataValidationArtifact)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered The Data Ingestion Function in the Training Pipeline")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the training and testing data ingestion artifact")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entered The Data Validation Function in the Training Pipeline")
            data_validation = DataValidation(data_ingestion_artifact= data_ingestion_artifact, data_validation_config= self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Got the training and testing data validation artifact")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
        except MyException as e:
            raise MyException(e, sys)