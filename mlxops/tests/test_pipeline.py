"""Testing pipeline module
"""

import unittest

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit

from mlxops.components import DataLoader, DataValidator,\
    ModelTrainer, ModelEvaluator, ArtifactPusher, DataFeatureMapper

from mlxops.pipeline import ModelTrainingPipeline, ScoringPipeline


class TestPipeline(unittest.TestCase):
    """Test the Preprocessing module pipelines
    """
    def setUp(self):
        self.url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"

    def test_train_pipeline(self):
        """Test full training pipeline
        """
        train_pipeline_arguments = {
            'data_loader': DataLoader.from_file(
                file_name=self.url, target='target', splitter=ShuffleSplit(n_splits=1, test_size=0.25)
            ),
            'data_validator': DataValidator(
                validator=IsolationForest(contamination=0.01)
            ),
            'feature_mapper': DataFeatureMapper.from_infered_pipeline(),
            'trainer': ModelTrainer(
                estimator=RandomForestClassifier(n_estimators=10, max_depth=3)
            ),
            'evaluator': ModelEvaluator(
                # TODO: the loaded base model is not using saved feature mapper
                base_model='mlxops/tests/test_artifacts',
                metrics=[roc_auc_score]
            ),
            'pusher': ArtifactPusher(model_serving_dir='testfolder'),
            "run_id": "random_forest_run"
        }

        train_pipeline = ModelTrainingPipeline(**train_pipeline_arguments)
        train_pipeline.run()
        # high level check to see that pipeline runs
        assert True

    def test_scoring_pipeline_from_file(self):
        """Test scoring pipeline"""
        data_loader = DataLoader.from_file(self.url)
        data_loader.data.drop('target', axis=1, inplace=True)
        scorer = ScoringPipeline.load_from_file("mlxops/tests/test_artifacts")
        scorer.run(data_loader)
        assert True
        assert scorer.mask.shape[0] == scorer.predictions.shape[0]        


if __name__ == '__main__':
    unittest.main()
