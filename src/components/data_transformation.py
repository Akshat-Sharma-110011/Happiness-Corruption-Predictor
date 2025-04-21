import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns={'_id': 'id'}, inplace=True)  # Only if you want to work with 'id'
            return df
        except Exception as e:
            raise MyException(e, sys)

    def get_data_preprocessor_object(self) -> Pipeline:
        logging.info("Entered get_data_preprocessor_object in DataTransformation")
        try:
            # Get features from schema
            numeric_features = self._schema_config['numerical_columns']
            cat_features = self._schema_config['categorical_columns']
            ss_features = self._schema_config.get('ss_feature', [])  # Features for StandardScaler

            # Initialize transformers
            power_transformer = PowerTransformer(method='yeo-johnson')
            std_scaler = StandardScaler()

            # Log columns information
            logging.info(f"Numeric Columns: {numeric_features}")
            logging.info(f"Categorical Columns: {cat_features}")
            logging.info(f"StandardScaler Features: {ss_features}")

            # Identify transformable numeric columns (excluding target and binary columns)
            transform_numeric_cols = [col for col in numeric_features
                                      if col != TARGET_COLUMN]

            # Use the ss_feature list from schema for StandardScaler columns
            standard_scale_cols = [col for col in transform_numeric_cols if col in ss_features]

            # Other numeric columns use PowerTransformer
            power_transform_cols = [col for col in transform_numeric_cols
                                    if col not in standard_scale_cols]

            # Create the column transformer
            transformers = []

            # Add PowerTransformer for most numeric columns
            if power_transform_cols:
                transformers.append(('power', power_transformer, power_transform_cols))
                logging.info(f"PowerTransformer will be applied to: {power_transform_cols}")

            # Add StandardScaler for columns specified in ss_feature
            if standard_scale_cols:
                transformers.append(('std', std_scaler, standard_scale_cols))
                logging.info(f"StandardScaler will be applied to: {standard_scale_cols}")

            # Create the preprocessor
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )

            # Create the final pipeline
            final_pipeline = Pipeline([('PREPROCESSOR', preprocessor)])
            logging.info("Final Pipeline ready")

            return final_pipeline
        except Exception as e:
            raise MyException(e, sys) from e

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered _create_dummy_columns in DataTransformation")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping columns specified in schema")
        drop_cols = self._schema_config.get('drop_columns', [])
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]

        columns_to_drop = [col for col in drop_cols if col in df.columns]
        if columns_to_drop:
            logging.info(f"Dropping columns: {columns_to_drop}")
            df = df.drop(columns_to_drop, axis=1)
        else:
            logging.info("No columns to drop")
        return df

    # Removed _map_gender_columns method as it's not needed for this dataset

    def _remove_outliers_iqr(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Remove outliers from DataFrame using IQR method for specified columns

        Parameters:
            df (pd.DataFrame): Input DataFrame
            columns (list): List of columns to check for outliers

        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        logging.info("Removing outliers using IQR method")
        result_df = df.copy()
        total_rows_start = result_df.shape[0]

        for col in columns:
            # Skip the target column and Year column (usually don't want to remove years as outliers)
            if col == TARGET_COLUMN or col == 'Year':
                logging.info(f"Skipping outlier removal for column '{col}'")
                continue

            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Create mask for non-outlier values
            mask = (result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)

            # Log outlier boundaries
            logging.info(f"Column '{col}' - IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")

            # Apply mask to DataFrame
            rows_before = result_df.shape[0]
            result_df = result_df[mask]

            # Log how many outliers were removed
            outliers_removed = rows_before - result_df.shape[0]
            logging.info(f"Removed {outliers_removed} outliers from column '{col}'")

        total_removed = total_rows_start - result_df.shape[0]
        logging.info(
            f"Total rows removed as outliers: {total_removed} ({(total_removed / total_rows_start * 100):.2f}% of data)")

        return result_df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Entered initiate_data_transformation in DataTransformation")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Drop ID columns if specified in schema
            train_df = self._drop_id_column(train_df)
            test_df = self._drop_id_column(test_df)
            logging.info("Checked and dropped columns as per schema")

            # Extract features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # No gender column mapping needed for this dataset

            # Get numeric columns for outlier removal from schema config
            numeric_cols = self._schema_config['numerical_columns']
            # Filter out the target column from numeric columns for outlier removal
            outlier_removal_cols = [col for col in numeric_cols if col != TARGET_COLUMN]

            logging.info(f"Columns for outlier removal: {outlier_removal_cols}")

            # Remove outliers from training data
            input_feature_train_df = self._remove_outliers_iqr(input_feature_train_df, outlier_removal_cols)
            # Target feature must match the rows in input_feature after outlier removal
            target_feature_train_df = target_feature_train_df.loc[input_feature_train_df.index]

            logging.info(f"After outlier removal - Training data shape: {input_feature_train_df.shape}")

            # Create dummy variables for categorical features
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)

            # Align columns
            input_feature_test_df = input_feature_test_df.reindex(columns=input_feature_train_df.columns, fill_value=0)

            logging.info("Created and aligned dummy columns for categorical features")

            logging.info("Created dummy columns for categorical features")
            logging.info(f"Train features shape after dummy creation: {input_feature_train_df.shape}")
            logging.info(f"Test features shape after dummy creation: {input_feature_test_df.shape}")

            # Get preprocessor object
            preprocessor = self.get_data_preprocessor_object()
            logging.info("Got the Preprocessor ready")

            # Transform data
            logging.info("Initializing transformation for training data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for test data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation completed end to end")

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_dir, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_dir, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_dir, test_arr)
            logging.info("Saved transformations in artifacts")

            logging.info("Data Transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_path=self.data_transformation_config.transformed_object_dir,
                transformed_train_file_path=self.data_transformation_config.transformed_train_dir,
                transformed_test_file_path=self.data_transformation_config.transformed_test_dir
            )
        except Exception as e:
            raise MyException(e, sys) from e