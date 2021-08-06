"""Model related Components of a full SDLC of a Machine Learning System"""

import logging
import pathlib
import shutil
import pandas

from .base import BasePipelineComponent


class ModelTrainer(BasePipelineComponent):
    """Model component of the training pipeline
    """
    def __init__(self, estimator=None):
        self.estimator = estimator

    def run(self, data_loader=None, feature_mapper=None, data_validator=None):
        """[summary]

        Args:
            data_loader ([type], optional): [description]. Defaults to None.
            feature_mapper ([type], optional): [description]. Defaults to None.
            data_validator ([type], optional): [description]. Defaults to None.
        """
        train_data, train_targets = data_loader.train_set
        train_data = train_data.loc[data_validator.trainset_valid]
        train_targets = train_targets.loc[data_validator.trainset_valid]
        # Should the feature mapper component live here?
        train_data_mapped = feature_mapper.transform(train_data)
        self.estimator.fit(train_data_mapped, train_targets)

    def predict(self, data=None, feature_mapper=None):
        """[summary]

        Args:
            data ([type], optional): [description]. Defaults to None.
            feature_mapper ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        data_mapped = feature_mapper.transform(data)
        return self.estimator.predict(data_mapped)

    @property
    def metadata(self):
        """Return model training metadata"""
        return {
            'estimator': self.estimator
        }


class ModelScore(BasePipelineComponent):
    """Score datasets with supplied model
    """
    def __init__(self, model=None):
        self.model = model
        self.predictions = {}

    def run(self, data_loader=None, feature_mapper=None, data_validator=None):
        """Score data with new and current prod model and compare results

        Returns:
            Scored samples
        """
        for set_name, data_set in data_loader.outputs.items():
            data_set = getattr(data_loader, set_name)[0]
            self.predictions[set_name] = self.model.predict(data_set)


class ModelEvaluator(BasePipelineComponent):
    """Compare metrics between trained model and current model in production
    """
    _BASE_MODEL_ATTR = 'base_model'
    _NEW_MODEL_ATTR = 'new_model'

    def __init__(self, base_model=None, metrics=None):
        self.base_model = base_model
        self.metrics = metrics
        self.evaluation_metrics = {}
        self.new_model = None
        self.push_model = None

    def run(self, data_loader=None, feature_mapper=None, data_validator=None, new_model=None):
        """Score data with new and current prod model and compare results

        Args:
            data_loader ([type], optional): [description]. Defaults to None.
            feature_mapper ([type], optional): [description]. Defaults to None.
            data_validator ([type], optional): [description]. Defaults to None.
        """
        self.new_model = new_model
        eval_data, eval_targets = data_loader.eval_set
        # eval_data = eval_data[data_validator.evalset_valid]
        new_model_predictions = self.new_model.predict(eval_data, feature_mapper)
        base_model_predictions = self.base_model.predict(eval_data, feature_mapper)

        for metric in self.metrics:
            self.evaluation_metrics[self._NEW_MODEL_ATTR] = metric(eval_targets, new_model_predictions)
            self.evaluation_metrics[self._BASE_MODEL_ATTR] = metric(eval_targets, base_model_predictions)

        self.push_model = self.evaluation_metrics[self._NEW_MODEL_ATTR] >\
            self.evaluation_metrics[self._BASE_MODEL_ATTR]

    @property
    def metadata(self):
        """Return evaluator metadata"""
        return {
            'push_model': self.push_model,
            'metrics': self.evaluation_metrics
        }
