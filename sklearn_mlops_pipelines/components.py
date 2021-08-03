"""Components for a full SDLC of a Machine Learning System"""

import logging
import pathlib
import shutil
import pandas

from base import BasePipelineComponent
from infered_feature_pipeline import InferedFeaturePipeline


class DataLoader(BasePipelineComponent):
    """Data loading component of the training pipeline"""
    _file_name = ""
    def __init__(self, data, target, preprocessors=[]):
        # transformer list should be stateless
        self.data = data
        self.target = target
        self.preprocessors = preprocessors
        if not isinstance(self.preprocessors, list):
            self.preprocessors = list(preprocessors)

    @classmethod
    def from_file(cls, file_name=None, *args, **kwargs):
        """Read in data from file and create DataLoader instance

        Args:
            file_name ([str]): File name of model data

        Returns:
            [DataLoader]: Instatiated object from file name
        """
        cls._file_name = file_name
        data = pandas.read_csv(cls._file_name)
        return cls(data, *args, **kwargs)

    def run(self):
        """Run component step"""
        if self.preprocessors:
            for preprocessor in self.preprocessors:
                self.data = preprocessor.transform(self.data)
        self._train_set, self._test_set, self._eval_set = [self.data.index.tolist()]*3  # temp hack while devving

    @property
    def outputs(self):
        """Return training and evaluation data as dictionary"""
        return {
            'train_set': self._train_set,
            'test_set': self._test_set,
            'eval_set': self._eval_set
        }

    @property
    def train_set(self):
        """Return training data"""
        return self.data.iloc[self._train_set, :], self.target.iloc[self._train_set]

    @property
    def test_set(self):
        """Return testing data"""
        return self.data.iloc[self._test_set, :], self.target.iloc[self._test_set]

    @property
    def eval_set(self):
        """Return evaluation data"""
        return self.data.iloc[self._eval_set, :], self.target.iloc[self._eval_set]

    @property
    def metadata(self):
        """return Data Loader metadata for training run"""
        return {
            'file_name': self._file_name,
            'datasets': self.outputs,
            # How will we handle preprocessor here
            'preprocessor': ""
        }


class DataInputValidator(BasePipelineComponent):
    """Data validation component of the training pipeline
    """
    _VALID_OBSERVATIONS_ATTR = 'valid_observations'
    _INVALID_OBSERVATIONS_ATTR = 'invalid_observations'

    def __init__(self, validator=None, data_loader=None):
        self.validator = validator
        self.data_loader = data_loader
        self.validness_indicator = {}

    def check_validity(self, data):
        """Run validator component and record validation indicator

        Args:
            data (pandas.DataFrame):

        Returns:
            (dict): dictionary containing the valid and invalid samples passed for training
        """
        # Below just for testing
        return self.data_loader

    def run(self, data_loader=None):
        """Run validator component and record validation indicator

        Returns:
            self
        """
        data_dict = self.data_loader.outputs
        for set_name, data_set in data_dict.items():
            self.validness_indicator[set_name] = data_set

    # create interface with self.data and self.train_set
    # we can think to join the two per property trainset_validness
    @property
    def trainset_valid(self):
        """Return list of valid observations"""
        return self.validness_indicator['train_set']#[self._VALID_OBSERVATIONS_ATTR]

    @property
    def trainset_invalid(self):
        """Return list of invalid observations"""
        return self.validness_indicator['train_set']#[self._INVALID_OBSERVATIONS_ATTR]

    @property
    def evalset_valid(self):
        """Return list of valid observations"""
        return self.validness_indicator['eval_set']#[self._VALID_OBSERVATIONS_ATTR]

    @property
    def evalset_invalid(self):
        """Return list of invalid observations"""
        return self.validness_indicator['eval_set']#[self._INVALID_OBSERVATIONS_ATTR]

    @property
    def metadata(self):
        """Return validator metadata"""
        return self.validness_indicator


class ModelTrainer(BasePipelineComponent):
    """Model component of the training pipeline"""

    def __init__(self, training_pipeline=None):
        self.training_pipeline = training_pipeline

    @classmethod
    def from_infered_pipeline(cls, estimator=None):
        # if it is an estimator infer pipeline
        return cls(
            InferedFeaturePipeline(estimator)
        )

    def run(self, data_loader=None, data_validator=None):
        """Fit model

        Returns:
            self
        """
        train_data, train_targets = data_loader.train_set
        # train_data = train_data.iloc[validator.traintrainset_valid, :]
        # train_targets = train_targets.iloc[validator.trainset_valid]
        # We should implement a tf.data.Dataset mapper before passing data to network
        # Should the feature mapper component live here?
        self.training_pipeline.fit(train_data, train_targets)
        return self

    @property
    def metadata(self):
        """Return model training metadata"""
        return {}


class ModelScore(BasePipelineComponent):
    def __init__(self, model=None):
        self.model = model
        self.predictions = {}

    def run(self, data_loader=None, data_validator=None):
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

    def __init__(self, base_model=None, new_model=None, metrics=None):
        self.base_model = base_model
        self.new_model = new_model
        self.metrics = metrics
        self.evaluation_metrics = {}
        self.push_model = None

    def run(self, data_loader=None, data_validator=None):
        """Score data with new and current prod model and compare results

        Returns:
            self
        """
        eval_data, eval_targets = data_loader.eval_set
        # eval_data = eval_data[data_validator.evalset_valid]
        new_model_predictions = self.new_model.predict(eval_data)
        base_model_predictions = self.base_model.predict(eval_data)

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


class ArtifactPusher(BasePipelineComponent):
    """Push all training artifacts if model performance better than prod model
    """

    def __init__(self, model_serving_dir=None):
        self.model_serving_dir = model_serving_dir

    def run(self):
        """Copy model training artifacts to model_data directory"""
        for content in MODEL_ARTIFACTS.glob('*.*'):
            shutil.copy(content, self.model_serving_dir)

    @property
    def metadata(self):
        """return metadata for artifact pusher"""
        return {
            'model_serving_dir': self.model_serving_dir
        }


if __name__ == '__main__':
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score


    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)
    target = dataframe.pop('target')
    # dataframe = dataframe[['sex', 'cp', 'fbs', 'oldpeak']]

    # data_loader = DataLoader(data=dataframe, target=target)
    data_loader = DataLoader.from_file(file_url, target=target)
    data_loader.run()
    data_validator = DataInputValidator(data_loader=data_loader)
    data_validator.run()


    # base_trainer = ModelTrainer(
    #     # model=pipeline
    #     # TODO: Infered pipeline should be in the Model trainer class if only estimator was supplied to model attr
    #     #       This will remove the wrapper around the estimator
    #     #       Q: How do we check if a pipeline only contains an estimator? If only estimator infer the feature mappers
    #     model=LogisticRegression()
    # )
    base_trainer = ModelTrainer.from_infered_pipeline(estimator=LogisticRegression())
    base_trainer.run(data_loader=data_loader, data_validator=data_validator)

    # new_trainer = ModelTrainer(
    #     model=RandomForestClassifier(n_estimators=50)
    # )
    # new_trainer.run(data_loader=data_loader, data_validator=data_validator)

    # # Model score
    # scorer = ModelScore(model=base_trainer.model)
    # scorer.run(data_loader=data_loader, data_validator=data_validator)

    # # Evaluator
    # evaluator = ModelEvaluator(base_model=base_trainer.model,
    #                            new_model=new_trainer.model,
    #                            metrics=[accuracy_score])
    # evaluator.run(data_loader=data_loader, data_validator=data_validator)
