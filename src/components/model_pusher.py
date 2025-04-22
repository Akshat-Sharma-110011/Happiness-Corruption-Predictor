import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.s3_estimator import Proj1Estimator

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifact = model_evaluation_artifact
        self.s3 = SimpleStorageService()
        self.proj1_estimator = Proj1Estimator(bucket_name=self.model_pusher_config.bucket_name,
                                              model_path=self.model_pusher_config.s3_bucket_key)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Initiating model_pusher")

            logging.info("Uploading new model to s3 bucket")
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_file_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_bucket_key)
            logging.info("Uploaded new model artifact to s3 bucket")
            logging.info(f"Model pusher artifact {model_pusher_artifact}")

            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e