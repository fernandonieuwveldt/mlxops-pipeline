"""Training pipeline"""

import logging
import pathlib

from .base import PipelineMixin

# Can have high level pipeline to test multiple models perhaps?>>
# Or Supply different estimators to the ModelTrainer component

class ModelTrainingPipeline(PipelineMixin):
    """Holds context throughout the pipeline"""

    def __init__(self, data_loader=None,
                       data_validator=None,
                       feature_mapper=None,
                       trainer=None,
                       evaluator=None,
                       pusher=None):
        self._logger = logging.getLogger(self.__class__.__name__)
        # initialise pipeline components
        self.data_loader = data_loader
        self.data_validator = data_validator
        self.feature_mapper = feature_mapper
        self.trainer = trainer
        self.evaluator = evaluator
        self.pusher = pusher

    def run(self):
        """Run all components of the training pipeline"""
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
            self.pusher.run()


class ScoringPipeline(PipelineMixin):
    """Scoring pipeline of new batches of samples
    """
    def __init__(self, data_loader=None, data_validator=None, scorer=None):
        self._logger = logging.getLogger(self.__class__.__name__)
        # initialise pipeline components
        self.data_loader = data_loader
        self.scorer = scorer
        self.data_validator = data_validator

    def run(self):
        """Run all components of the scoring pipeline"""
        self.data_loader.run()
        self.data_validator.run(data_loader=self.data_loader)
        results = self.scorer.run(data_loader=self.data_loader(split_percentage=0),  # for scoring train and eval are the same
                                  data_validator=self.data_validator)
