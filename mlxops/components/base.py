"""Base class for all Pipeline Components"""

from abc import ABC, abstractmethod
import pickle
import inspect


class BaseComponent(ABC):
    """Base Class for a training pipeline component. Each step in ML life cycle will inherit from this Base class.
    Base clas methods mainly contains saving and loading of pipeline components artifacts .
    """

    _type = "component"

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the pipeline component"""
        return self

    @property
    @abstractmethod
    def metadata(self):
        """Return any metadata that was recorded during the component run"""

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

    def set_local_components(self, local_variables):
        for name, var in local_variables.items():
            if hasattr(var, 'run'):
                self.__dict__.update({name: var})

    def __repr__(self):
        return f"Completed {self.__class__.__name__} component run"
