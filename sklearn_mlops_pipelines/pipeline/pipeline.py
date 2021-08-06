"""Training pipeline"""

import logging
import pathlib


from components import DataLoader, DataInputValidator, ModelTrainer, ModelEvaluator, ArtifactPusher, ModelScore
from base import PipelineMixin


class ModelTrainingPipeline(PipelineMixin):
    """Holds context throughout the pipeline"""

    def __init__(self, data_loader=None,
                       data_validator=None,
                       trainer=None,
                       model_evaluator=None,
                       pusher=None):
        self._logger = logging.getLogger(self.__class__.__name__)
        # initialise pipeline components
        self.data_loader = data_loader
        self.data_validator = data_validator
        self.trainer = trainer
        self.evaluator = evaluator
        self.pusher = pusher

    def run(self):
        """Run all components of the training pipeline"""
        self.data_loader.run()
        self.data_validator.run(data_loader=self.data_loader)
        self.trainer.run(data_loader=self.data_loader, data_validator=self.data_validator)
        self.evaluator.run(data_loader=self.data_loader, data_validator=self.validator, new_model=self.trainer.model)
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


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
target = dataframe.pop('target')
data = dataframe[['sex', 'cp', 'fbs', 'oldpeak']]


scoring_pipeline_arguments = {
    'data_loader': DataLoader(data=data, target='target'),
    'data_validator': DataInputValidator(),
    'scorer': ModelScore(trained_model)
}
scoring_pipeline = ScoringPipeline(**scoring_pipeline_arguments)


train_pipeline_arguments = {
    'data_loader': DataLoader(file_name='train_data.csv'),
    'data_validator': DataInputValidator(),
    'trainer': ModelTrainer(),
    'evaluator': ModelEvaluator(baseline_model=SomeModel),
    'pusher': ArtifactPusher(model_serving_dir),
}

train_pipeline = ModelTrainingPipeline(**train_pipeline_arguments)
