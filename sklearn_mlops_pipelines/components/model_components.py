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
        """Retrieve data from data_loader and the validness mask from data_validator and fit estimator

        Args:
            data_loader ([DataLoader]): data loading component holding data and dataset splits.
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
            data_validator ([DataInputValidator]): Data input validator.
        """
        train_data, train_targets = data_loader.train_set
        train_data = train_data.loc[data_validator.trainset_valid]
        train_targets = train_targets.loc[data_validator.trainset_valid]
        train_data_mapped = feature_mapper.transform(train_data)
        self.estimator.fit(train_data_mapped, train_targets)

    def predict(self, data=None, feature_mapper=None):
        """Apply feature mapper to map raw feature to estimator ready array and apply fitted estimator.

        Args:
            data ([pandas.DataFrame]): Raw feature data
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.

        Returns:
            [array]: array of predictions
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
        """Score data with new and current production(or best) model and compare results.

        Args:
            data ([pandas.DataFrame]): Raw feature data
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
            data_validator ([DataInputValidator]): Data input validator.
        """
        for set_name, data_set in data_loader.outputs.items():
            data_set = getattr(data_loader, set_name)[0]
            self.predictions[set_name] = self.model.predict(data_set)
        return self


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
            data ([pandas.DataFrame]): Raw feature data
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
            data_validator ([DataInputValidator]): Data input validator.
        """
        self.new_model = new_model
        eval_data, eval_targets = data_loader.eval_set
        eval_data = eval_data[data_validator.evalset_valid]
        new_model_predictions = self.new_model.predict(eval_data, feature_mapper)
        base_model_predictions = self.base_model.predict(eval_data, feature_mapper)
        import pdb
        pdb.set_trace()
        for metric in self.metrics:
            self.evaluation_metrics[self._NEW_MODEL_ATTR] = metric(eval_targets, new_model_predictions)
            self.evaluation_metrics[self._BASE_MODEL_ATTR] = metric(eval_targets, base_model_predictions)

        self.push_model = self.evaluation_metrics[self._NEW_MODEL_ATTR] >\
            self.evaluation_metrics[self._BASE_MODEL_ATTR]
        return self

    @property
    def metadata(self):
        """Return evaluator metadata"""
        return {
            'push_model': self.push_model,
            'metrics': self.evaluation_metrics
        }
