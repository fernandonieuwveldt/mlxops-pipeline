"""Components for a full SDLC of a Machine Learning System"""

import logging
import pathlib
import shutil
import pandas

from .base import BasePipelineComponent
from .infered_feature_pipeline import InferedFeaturePipeline


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
        # Should this be the prefered way to get the data ito metadata to point to?
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
        """return DataLoader metadata for training run"""
        return {
            'file_name': self._file_name,
            'preprocessor': self.preprocessor,
            **self.outputs
        }


class DataFeatureMapper(BasePipelineComponent):
    """Feature processing pipeline

    Args:
        BasePipelineComponent ([type]): [description]
    """
    def __init__(self, feature_pipeline=None):
        self.feature_pipeline = feature_pipeline

    @classmethod
    def from_infered_pipeline(cls, *args, **kwargs):
        """Use infered pipeline

        Returns:
            ModelTrainer: ModelTrainer initialised object with infered pipeline
        """
        return cls(
            InferedFeaturePipeline()
        )

    def run(self, data_loader=None):
        """Fit model

        Returns:
            self
        """
        train_data, train_targets = data_loader.train_set
        self.feature_pipeline.fit(train_data, train_targets)

    @property
    def metadata(self):
        """return DataFeatureMapper metadata for training run"""
        return {
            'feature_pipeline': self.feature_pipeline
        }


class DataInputValidator(BasePipelineComponent):
    """Data validation component of the training pipeline
    """
    _VALID_OBSERVATIONS_ATTR = 'valid_observations'
    _INVALID_OBSERVATIONS_ATTR = 'invalid_observations'

    def __init__(self, validator=None):
        self.validator = validator
        self.validness_indicator = {}

    def check_validity(self, data, feature_mapper):
        """Run validator component and record validation indicator

        Args:
            data (pandas.DataFrame):

        Returns:
            (dict): dictionary containing the valid and invalid samples passed for training
        """
        data_set_mapped = feature_mapper.feature_pipeline.transform(data)
        state = self.validator.predict(data_set_mapped)
        mask = state != -1
        return mask

    def run(self, data_loader=None, feature_mapper=None):
        """Run validator component and record validation indicator

        Returns:
            self
        """
        # validator can be applied on all sets to record metadata,
        # but only trainset can use the outlier mask
        train_data = getattr(data_loader, 'train_set')[0]
        data_set_mapped = feature_mapper.feature_pipeline.transform(train_data)
        self.validator.fit(data_set_mapped)
        for set_name, data_set in data_loader.outputs.items():
            data_set = getattr(data_loader, set_name)[0]
            self.validness_indicator[set_name] = self.check_validity(data_set, feature_mapper)

    @property
    def trainset_valid(self):
        """Return list of valid observations"""
        return self.validness_indicator['train_set']

    @property
    def trainset_invalid(self):
        """Return list of invalid observations"""
        return ~self.validness_indicator['train_set']

    @property
    def evalset_valid(self):
        """Return list of valid observations"""
        return self.validness_indicator['eval_set']

    @property
    def evalset_invalid(self):
        """Return list of invalid observations"""
        return ~self.validness_indicator['eval_set']

    @property
    def metadata(self):
        """Return validator metadata"""
        return {
            'validator': self.validator,
            'validness_indicator': self.va
        }
