import logging
import pathlib
import shutil
import pandas

from .base import BasePipelineComponent


MODEL_ARTIFACTS="."


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
