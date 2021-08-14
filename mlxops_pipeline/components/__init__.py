from .data_components import DataLoader, DataFeatureMapper, DataValidator
from .model_components import ModelTrainer, ModelEvaluator, ModelScore
from .serving_components import ArtifactPusher


_COMPONENT_MAPPER = {
    "DataLoader": DataLoader,
    "DataValidator": DataValidator,
    "DataFeatureMapper": DataFeatureMapper,
    "ModelTrainer": ModelTrainer,
    "ModelEvaluator": ModelEvaluator,
    "ModelScore": ModelScore,
    "ArtifactPusher": ArtifactPusher
}
