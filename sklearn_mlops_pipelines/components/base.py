from abc import ABC, abstractmethod


class BasePipelineComponent:
    """Base Class for a training pipeline component"""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the pipeline component"""

    @property
    @abstractmethod
    def metadata(self):
        """Return any metadata that was recorded during the component run"""

    @abstractmethod
    def save(self):
        """Save metadata to disk
        """

    @abstractmethod   
    def load(self):
        """Load saved metadata
        """
