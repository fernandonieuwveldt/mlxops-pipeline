"""Model related Components of a full SDLC of a Machine Learning System"""

import logging
import pathlib
import shutil
import pandas

from .base import BasePipelineComponent


class ModelTrainer(BasePipelineComponent):
    """
    Model component of the training pipeline. ModelTrainer is the step 4 in the Pipeline and needs
    a runned DataLoader, DataFeatureMapper and DataValidator to fit estimator. 

    Examples
    --------
    >>> trainer = ModelTrainer(
        estimator=LogisticRegression()
    )
    >>> trainer.run(data_loader=data_loader, feature_mapper=feature_mapper, data_validator=data_validator)
    """
    def __init__(self, estimator=None):
        self.estimator = estimator

    def run(self, data_loader=None, feature_mapper=None, data_validator=None):
        """Retrieve data from data_loader and the validness mask from data_validator and fit estimator

        Args:
            data_loader ([DataLoader]): data loading component holding data and dataset splits.
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
            data_validator ([DataValidator]): Data input validator.
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


class ModelEvaluator(BasePipelineComponent):
    """Compare metrics between trained model and current model in production

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> evaluator = ModelEvaluator(
        base_model="base_model_folder", new_model=new_trainer, metrics=[accuracy_score]
    )
    >>> evaluator.run(data_loader=data_loader, feature_mapper=feature_mapper, data_validator=data_validator)
    """
    _BASE_MODEL_ATTR = 'base_model'
    _NEW_MODEL_ATTR = 'new_model'

    def __init__(self, base_model=None, metrics=None):
        # TODO: base_model option can also be a runned ModelTrainer object
        self.base_model = base_model
        if self.base_model is not None:
            self.load_base_model_artifacts(self.base_model)
        self.metrics = metrics
        self.evaluation_metrics = {}
        self.push_model = None

    def load_base_model_artifacts(self, base_model_folder):
        """Load base model artifacts. Persisted DataFeatureMapper and ModelTrainer should be loaded.

        Args:
            base_model_folder ([str]): directory of saved model artifacts
        """
        from mlxops_pipeline.components import DataFeatureMapper
        self.base_model_feature_mapper = DataFeatureMapper.load(base_model_folder)
        self.base_model_estimator = ModelTrainer.load(base_model_folder)

    def run(self, data_loader=None, feature_mapper=None, data_validator=None, new_model=None):
        """Score data with new and current prod model and compare results

        Args:
            data ([pandas.DataFrame]): Raw feature data
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
            data_validator ([DataValidator]): Data input validator.
        """
        eval_data, eval_targets = data_loader.eval_set
        # create properties in validator
        eval_data = eval_data[data_validator.evalset_valid]
        eval_targets = eval_targets[data_validator.evalset_valid]
        new_model_predictions = new_model.predict(eval_data, feature_mapper)
        # this is not the correct way to do it. Fix later
        if self.base_model:
            base_model_predictions = self.base_model_estimator.predict(eval_data, self.base_model_feature_mapper)

        for metric in self.metrics:
            self.evaluation_metrics[self._NEW_MODEL_ATTR] = metric(eval_targets, new_model_predictions)
            # this is not the correct way to do it. Fix later
            if self.base_model:
                self.evaluation_metrics[self._BASE_MODEL_ATTR] = metric(eval_targets, base_model_predictions)
            else:
                self.evaluation_metrics[self._BASE_MODEL_ATTR] = 0.0

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


class ModelScore(BasePipelineComponent):
    """Score datasets with supplied model
    """
    def __init__(self, model=None):
        self.model = model
        self.predictions = {}

    def run(self, data_loader=None, feature_mapper=None, data_validator=None):
        """Score data with new and current production(or best) model and compare results.

        Args:
            data_loader ([DataLoader]): data loading component holding data and dataset splits.
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
            data_validator ([DataValidator]): Data input validator.
        """
        for set_name, data_set in data_loader.outputs.items():
            data_set = getattr(data_loader, set_name)[0]
            self.predictions[set_name] = self.model.predict(data_set)
        return self
