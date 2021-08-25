"""Training pipeline"""

import logging
import pathlib

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
        self.feature_mapper.run(data_loader=self.data_loader)
        self.data_validator.run(
            data_loader=self.data_loader, feature_mapper=self.feature_mapper
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
    """Scoring pipeline of new batches of samples
    """
    def __init__(self, data_loader=None, data_validator=None, scorer=None):
        # initialise pipeline components
        self.data_loader = data_loader
        self.scorer = scorer
        self.data_validator = data_validator

    def run(self):
        """Run all components of the scoring pipeline"""
        self.data_loader.run()
        self.data_validator.run(data_loader=self.data_loader)
        results = self.scorer.run(data_loader=self.data_loader(),
                                  data_validator=self.data_validator)
