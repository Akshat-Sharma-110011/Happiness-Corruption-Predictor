import os
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.exception import MyException
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            my_data = Proj1Data()
            df = my_data.export_collection_as_dataframe(collection_name= self.data_ingestion_config.collection_name)

            logging.info(f"Shape of the dataframe: {df.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            directory = os.path.dirname(feature_store_file_path)
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Writing the raw data to the feature store: {feature_store_file_path}")
            df.to_csv(feature_store_file_path, index=False, header=True)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame):
        try:
            logging.info("Splitting data into train and test")
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed Train and Test Split")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Writing the train and test split to the directory {dir_path}")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Finished splitting data into train and test")
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion")
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)
            logging.info("Finished data ingestion")
            logging.info("Exiting data ingestion methods")

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e