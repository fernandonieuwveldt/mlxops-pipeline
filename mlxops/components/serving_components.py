import logging
import pathlib
import shutil
import pandas

from .base import BaseComponent

MODEL_DEV_DIR="dev"
# If dev model is better than current release(best) model
# Copy artifacts of dev model to release directory
# Move previous release model to prod_previous?
MODEL_RELEASE_DIR = "release"
MODEL_CURRENT_PROD = 'prod'
MODEL_PREV_PROD = 'pre_prod'


class ArtifactPusher(BaseComponent):
    """Push all training artifacts if model performance better than prod model
    """
    ARTIFACT_DIR = 'saved_models'

    def __init__(self, model_serving_dir=None, run_id=None):
        self.model_serving_dir = model_serving_dir
        self.run_id = run_id

    def run(self, run_id=None):
        """Copy model training artifacts to model_data directory"""
        run_id_artifact_dir = pathlib.Path(f"{MODEL_DEV_DIR}/{run_id}")
        for content in run_id_artifact_dir.glob('*.*'):
            shutil.copy(content, self.model_serving_dir)

    @property
    def metadata(self):
        """return metadata for artifact pusher"""
        return {
            'model_serving_dir': self.model_serving_dir
        }
