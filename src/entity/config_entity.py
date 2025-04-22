import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

train_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(train_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_DIR_NAME, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_DIR_NAME, TEST_FILE_NAME)
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(train_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    data_validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(train_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_object_dir: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DIR)
    transformed_train_dir: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_PREPROCESSED_DATA_DIR, TRAIN_FILE_NAME.replace(".csv", "npy"))
    transformed_test_dir: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_PREPROCESSED_DATA_DIR, TEST_FILE_NAME.replace(".csv", "npy"))

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(train_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    model_trainer_trained_model_dir: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    model_trainer_config_file_path: str = MODEL_TRAINER_CONFIG_FILE_PATH
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    _n_estimators: int = MODEL_N_ESTIMATORS
    _max_depth: int = MODEL_MAX_DEPTH
    _learning_rate: float = MODEL_LEARNING_RATE
    _subsample: float = MODEL_SUBSAMPLE
    _colsample_bytree: float = MODEL_COLSAMPLE_BYTREE

@dataclass
class ModelEvaluationConfig:
    changed_evaluation_threshold: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_bucket_key: str = MODEL_PUSHER_S3_KEY

@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_bucket_key: str = MODEL_PUSHER_S3_KEY