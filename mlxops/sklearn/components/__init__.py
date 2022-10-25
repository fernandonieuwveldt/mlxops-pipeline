from .data_components import DataLoader, DataFeatureMapper, DataValidator
from .model_components import ModelTrainer, ModelEvaluator
from .serving_components import ArtifactPusher, CreateInferencePipeline


_COMPONENT_MAPPER = {
    "DataLoader": DataLoader,
    "DataValidator": DataValidator,
    "DataFeatureMapper": DataFeatureMapper,
    "ModelTrainer": ModelTrainer,
    "ModelEvaluator": ModelEvaluator,
    "ArtifactPusher": ArtifactPusher,
    "CreateInferencePipeline": CreateInferencePipeline
}
