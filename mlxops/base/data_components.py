from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractproperty


class BaseDataLoader(metaclass=ABCMeta):
    """
    Base Data loading component of the training pipeline
    """

    @abstractclassmethod
    def from_file(cls, file_name=None, *args, **kwargs):
        """Read in data from file and create DataLoader instance

        Args:
            file_name ([str]): File name of model data

        Returns:
            [DataLoader]: Instatiated object from file name
        """

    @abstractmethod
    def run(self):
        """Apply preprocessors if supplied. Split data into train and test splits using splitter

        Returns:
            [DataLoader]: self
        """

    @abstractproperty
    def outputs(self):
        """Return training and evaluation data as dictionary"""
        return {
            'train_set': self._train_set,
            'test_set': self._test_set,
            'eval_set': self._eval_set
        }

    @abstractproperty
    def train_set(self):
        """Return training data"""

    @abstractproperty
    def test_set(self):
        """Return testing data"""

    @abstractproperty
    def eval_set(self):
        """Return evaluation data"""

    @abstractproperty
    def metadata(self):
        """return DataLoader metadata for training run"""
