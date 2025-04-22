from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class RegressionMetricsArtifact:
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: RegressionMetricsArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_score: float
    s3_model_file_path: str
    trained_model_file_path: str

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str