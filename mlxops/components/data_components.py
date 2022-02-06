"""Components for a full SDLC of a Machine Learning System"""

import logging
import pathlib
import shutil
import pandas
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline

from .base import BaseComponent
from .infered_feature_pipeline import InferedFeaturePipeline


class DataLoader(BaseComponent):
    """
    Data loading component of the training pipeline

    Examples
    --------
    >>> data_loader = DataLoader.from_file(
        file_url, target='target', splitter=ShuffleSplit(n_splits=1, test_size=0.1)
    )
    >>> data_loader.run()
    """
    N_SPLITS = 1
    SPLIT_RATIOS = {
        'train_ratio': 0.75,
        'eval_ratio': 0.15,
        'test_ratio': 0.10
    }
    def __init__(self, data, target=None, feature_transformer=[], splitter=None, split_ratios=None):
        if target is not None:
            self.target = data.pop(target)
        self.data = data
        self.feature_transformer = feature_transformer
        if self.feature_transformer:
            if isinstance(self.feature_transformer, list):
                self.feature_transformer = make_pipeline(*self.feature_transformer)
            else:
                self.feature_transformer = make_pipeline(*[self.feature_transformer])
        if splitter is not None:
            if not hasattr(splitter, "get_n_splits"):
                raise("splitter should have get_n_splits method")
        self.splitter = splitter
        if split_ratios is None:
            self.split_ratios = self.SPLIT_RATIOS
        # preprocessors should be stateless
        self._train_set = None
        self._test_set = None
        self._eval_set = None

    # DataLoader should be instantiated by loading data and preferable for
    # pickling/unpickling. See __getstate__ and setstate__
    @classmethod
    def from_file(cls, file_name=None, *args, **kwargs):
        """Read in data from file and create DataLoader instance

        Args:
            file_name ([str]): File name of model data

        Returns:
            [DataLoader]: Instatiated object from file name
        """
        cls.file_name = file_name
        data = pandas.read_csv(cls.file_name)
        return cls(data, *args, **kwargs)

    def split_data(self, X, y, splitter):
        """
        Split data in train and test set
        """
        splitter.get_n_splits(
            X=X, y=y,
        )
        return next(
            splitter.split(X=X, y=y)
        )

    def create_sets(self):
        """
        Create train, eval and test sets
        """
        train_splitter = self.splitter(n_splits=self.N_SPLITS, test_size=1-self.split_ratios['train_ratio'])
        self._train_set, self._test_set = self.split_data(self.data, self.target, train_splitter)
        # use test_set property to split again
        _test_data, _test_targets = self.test_set
        _test_split_ratio = self.split_ratios['test_ratio'] /\
                            (self.split_ratios['test_ratio'] + self.split_ratios['eval_ratio'])
        test_eval_splitter = self.splitter(n_splits=self.N_SPLITS, test_size=_test_split_ratio)
        self._eval_set, self._test_set = self.split_data(_test_data, _test_targets, test_eval_splitter)

    def run(self):
        """Apply preprocessors if supplied. Split data into train and test splits using splitter

        Returns:
            [DataLoader]: self
        """
        if self.feature_transformer:
            # Should the preprocessor run after the data was split?
            # And than run preprocessor separately on each split?
            self.data = self.feature_transformer.transform(self.data)

        if all([self.splitter is not None,
                self.target is not None]):
            self.create_sets()
        return self

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
        if not hasattr(self, 'file_name'):
            self.file_name = None

        return {
            'file_name': self.file_name,
            'feature_transformer': self.feature_transformer,
            **self.outputs
        }

    def __getstate__(self):
        """Update and delete data from state
        """
        state = self.__dict__.copy()
        # add file_name to state
        if hasattr(self, "file_name"):
            state['file_name'] = self.file_name
        # Remove the unpicklable entries.
        del state['data']
        return state

    def __setstate__(self, state):
        """Restore instance attribute by loading data from file
        """
        self.__dict__.update(state)
        # Restore data attribute by loading from file
        if 'file_name' in self.__dict__:
            import pandas
            self.data = pandas.read_csv(self.file_name)


class DataFeatureMapper(BaseComponent, TransformerMixin):
    """Feature processing pipeline. Apply Feature mapper for example Normalization for
    numeric features and OneHotEncoder for categoric features. Mapper here is stateful.

    Examples
    --------
    >>> feature_mapper = DataFeatureMapper.from_infered_pipeline()
    >>> # run takes DataLoader object as input
    >>> feature_mapper.run(data_loader=data_loader)
    """
    def __init__(self, feature_pipeline=None):
        self.feature_pipeline = feature_pipeline

    @classmethod
    def from_infered_pipeline(cls, *args, **kwargs):
        """Use infered pipeline

        Returns:
            DataFeatureMapper: Initialised object with InferedFeaturePipeline
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
        self.fit(train_data)
        return self

    def fit(self, X, y=None):
        """Wrap feature pipeline and fit pipeline

        Args:
            X ([pandas.DataFrame]): Features DataFrame
            y ([pandas.Series], optional): Targets

        Returns:
            [DataFeatureMapper]: self
        """
        self.feature_pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        """Transform X with fitted feature_pipeline

        Args:
            X ([pandas.DataFrame]): Features DataFrame

        Returns:
            [array]: array of transformed/mapped features for estimator input 
        """
        return self.feature_pipeline.transform(X)
        
    @property
    def metadata(self):
        """return DataFeatureMapper metadata for training run"""
        return {
            'feature_pipeline': self.feature_pipeline
        }


class DataValidator(BaseComponent):
    """Data validation component of the training pipeline

    Examples
    --------
    >>> data_validator = DataValidator(
        validator=IsolationForest(contamination=0.01)
    )
    >>> data_validator.run(data_loader=data_loader, feature_mapper=feature_mapper)
    """
    def __init__(self, validator=None):
        self.validator = validator
        self.validness_indicator = {}

    def check_validity(self, data, feature_mapper):
        """Run validator component and record validation indicator

        Args:
            data ([pandas.DataFrame]): Raw feature data
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.

        Returns:
            [array]: Boolean mask for valid and invalid samples.
        """
        data_set_mapped = feature_mapper.transform(data)
        state = self.validator.predict(data_set_mapped)
        mask = state != -1
        return mask

    def _run_with_validator(self, data_loader=None, feature_mapper=None):
        """Run data validator component and record validation indicator by applying validator

        Args:
            data_loader ([DataLoader]): data loading component holding data and dataset splits.
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
        """
        # validator can be applied on all sets to record metadata,
        # but only trainset can use the outlier mask
        train_data = getattr(data_loader, 'train_set')[0]
        data_set_mapped = feature_mapper.transform(train_data)
        self.validator.fit(data_set_mapped)

        for set_name, data_set in data_loader.outputs.items():
            data_set = getattr(data_loader, set_name)[0]
            self.validness_indicator[set_name] = self.check_validity(data_set, feature_mapper)

    def run(self, data_loader=None, feature_mapper=None):
        """Run component. If validator is None the validness_indicator mask is the same as the train/test split mask
        from the data_loader.

        Args:
            data_loader ([DataLoader]): data loading component holding data and dataset splits.
            feature_mapper ([DataFeatureMapper]): Feature transformer steps of a sklearn Pipeline.
        """
        self.validness_indicator = {
            **data_loader.outputs
        }

        if self.validator is not None:
            self._run_with_validator(data_loader, feature_mapper)
        return self

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
            'validness_indicator': self.validness_indicator
        }
