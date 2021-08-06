"""Mixin for all pipelines"""


class PipelineMixin:
    """Mixin class for pipelines"""

    @property
    def model_training_metadata(self):
        """Return metadata of all runned pipeline components"""
        return {component.__class__.__name__: component.metadata
                for _, component in self.__dict__.items() if hasattr(component, 'run')
        }
