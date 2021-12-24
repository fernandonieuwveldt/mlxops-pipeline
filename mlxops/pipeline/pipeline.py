"""Training pipeline"""

import logging
import pathlib

import mlxops
from .base import BasePipeline
from mlxops.components import ArtifactPusher


class ModelTrainingPipeline(BasePipeline):
    """Holds context throughout the pipeline"""

    def __init__(self, data_loader=None,
                       data_validator=None,
                       feature_mapper=None,
                       trainer=None,
                       evaluator=None,
                       pusher=None,
                       run_id=None):
        # initialise pipeline components
        self.data_loader = data_loader
        self.data_validator = data_validator
        self.feature_mapper = feature_mapper
        self.trainer = trainer
        self.evaluator = evaluator
        self.pusher = pusher
        self.run_id = run_id

    def run(self):
        """Run all components of the training pipeline"""
        # Should each component keep a reference to previous components in the pipeline?
        self.data_loader.run()
        self.feature_mapper.run(
            data_loader=self.data_loader
        )
        self.data_validator.run(
            data_loader=self.data_loader,
            feature_mapper=self.feature_mapper
        )
        self.trainer.run(
            data_loader=self.data_loader,
            feature_mapper=self.feature_mapper,
            data_validator=self.data_validator
        )
        self.evaluator.run(
            data_loader=self.data_loader,
            feature_mapper=self.feature_mapper,
            data_validator=self.data_validator,
            new_model=self.trainer
        )
        if self.evaluator.push_model:
            self.pusher.run(self.run_id)


class ScoringPipeline(BasePipeline):
    """Score datasets with supplied model
    """
    def __init__(self, model, data_loader, data_validator, feature_mapper):
        self.model = model
        self.data_loader = data_loader
        self.data_validator = data_validator
        self.feature_mapper = feature_mapper
        self.mask = None
        self.predictions = {}

    @classmethod
    def load_from_file(cls, folder=None):
        """
        Load train artifacts from folder
        """
        # loading should be handled by loading all components to one object
        model = mlxops.saved_model.load_component(f"{folder}/ModelTrainer.pkl")
        data_loader = mlxops.saved_model.load_component(f"{folder}/DataLoader.pkl")
        feature_mapper = mlxops.saved_model.load_component(f"{folder}/DataFeatureMapper.pkl")
        data_validator = mlxops.saved_model.load_component(f"{folder}/DataValidator.pkl")
        return cls(
            model = model,
            data_loader = data_loader,
            data_validator = data_validator,
            feature_mapper = feature_mapper
            )

    def run(self, data=None):
        """Score data with supplied model.

        Args:
            data ([pandas.DataFrame]): data to be scored
        """
        # apply stateless feature transformer if applied
        if self.data_loader.feature_transformer:
            data = self.data_loader.feature_transformer.transform(data)
        # get mask if validator was run
        if self.data_validator.validator is not None:
            self.mask = self.data_validator.check_validity(data, self.feature_mapper)
        # we will predict on all samples. Use mask to filter after scoring
        self.predictions = self.model.predict(data, self.feature_mapper)
        return self
