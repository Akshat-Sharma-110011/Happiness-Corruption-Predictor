import sys
import os
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class Proj1Data:
    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(DATABASE_NAME)
        except MyException as err:
            raise MyException(err, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.database[database_name][collection_name]

            print("Fetching data from collection {}".format(collection_name))
            df = pd.DataFrame(list(collection.find()))
            print("Exported dataframe shape {}".format(df.shape))
            if "id" in df.columns.tolist():
                df.drop(["id"], axis=1, inplace=True)
            df.replace("na", np.nan, inplace=True)
            return df
        except MyException as err:
            raise MyException(err, sys)