from .data_components import DataLoader, DataFeatureMapper, DataInputValidator
from .model_components import ModelTrainer, ModelEvaluator, ModelScore
from .serving_components import ArtifactPusher

_COMPONENT_MAPPER = {
    "DataLoader": DataLoader,
    "DataInputValidator": DataInputValidator,
    "DataFeatureMapper": DataFeatureMapper,
    "ModelTrainer": ModelTrainer,
    "ModelEvaluator": ModelEvaluator,
    "ModelScore": ModelScore,
    "ArtifactPusher": ArtifactPusher
}
