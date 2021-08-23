"""Base class for all Pipeline Components"""

from abc import ABC, abstractmethod
import pickle
import inspect


class BasePipelineComponent(ABC):
    """Base Class for a training pipeline component. Each step in ML life cycle will inherit from this Base class.
    Base clas methods mainly contains saving and loading of pipeline components artifacts .
    """
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the pipeline component"""

    @property
    @abstractmethod
    def metadata(self):
        """Return any metadata that was recorded during the component run"""

    def save(self, artifact_dir=None):
        """Save metadata to disk

        Args:
            artifact_dir ([str]): location of artifact
        """
        pickle.dump(
            self, open(f"{artifact_dir}/{self.__class__.__name__}.pkl", 'wb')
        )

    @classmethod
    def load(cls, artifact_dir=None):
        """Load saved metadata

        Args:
            artifact_dir ([str]): location of artifact

        Returns:
            [type]: [description]
        """
        return pickle.load(
            open(f"{artifact_dir}/{cls.__name__}.pkl", 'rb')
        )

    @classmethod
    def get_init_args(cls):
        """Get constructor names"""
        # get the contructor arguments of the component
        init_signature = inspect.signature(cls.__init__)
        return sorted(
            [p.name for p in init_signature.parameters.values() if p.name != 'self']
        )

    def get_init_values(self):
        """
        Get instance attribute values
        """
        return {
            key: getattr(self, key) for key in self.get_init_args() 
        }
