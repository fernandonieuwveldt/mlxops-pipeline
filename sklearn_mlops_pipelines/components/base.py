"""Base class for all Pipeline Components"""

from abc import ABC, abstractmethod
import pickle


class BasePipelineComponent:
    """Base Class for a training pipeline component"""

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
            open(f"{artifact_dir}/{cls.__class__.__name__}", 'rb')
        )
   
    @classmethod
    def from_config(cls, artifact_dir=None):
        pass

    def get_config(self):
        return self.metadata
