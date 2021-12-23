"""Base class for all pipelines"""

import pathlib
import json
import pickle

from mlxops.components import _COMPONENT_MAPPER


class BasePipeline:
    """Base class for all Pipeline. Contains metadata registring and saving final component state
    during pipeline"""

    _type = "pipeline"

    @property
    def model_training_metadata(self):
        """Return metadata of all runned pipeline components
        """
        return {component.__class__.__name__: component.metadata
                for _, component in self.__dict__.items() if hasattr(component, 'run')
        }

    def __repr__(self):
        return f"Completed {self.__class__.__name__} pipeline run"
