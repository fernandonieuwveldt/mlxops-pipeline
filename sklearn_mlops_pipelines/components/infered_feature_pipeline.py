"""Infered Feature Pipeline for auto solution"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURE_TRANSFORMER = 'numerical'
CATEGORICAL_FEATURE_TRANSFORMER = 'categorical'


def infered_pipeline(X=None):
    """Set estimator to be chain to the inferered pipeline

    Args:
        estimator ([sklearn estimator]): estimator that conforms to sklearn estimator type
        X ([pandas.DataFrame]): DataFrame containing feature data

    Returns:
        [pipeline.Pipeline]: infer pipeline and append estimator at the end of pipeline
    """
    # TODO: extend on types
    categorical_features = X.columns[X.dtypes == 'object'].tolist()
    numerical_features = X.columns[X.dtypes != 'object'].tolist()
    numerical_transformer_step = (NUMERIC_FEATURE_TRANSFORMER, Normalizer(), numerical_features)
    categorical_transformer_step = (CATEGORICAL_FEATURE_TRANSFORMER, OneHotEncoder(handle_unknown="ignore"), categorical_features)
    return Pipeline([
        ('infered_feature_pipeline', ColumnTransformer([numerical_transformer_step, categorical_transformer_step]))
    ])


class InferedFeaturePipeline(BaseEstimator, TransformerMixin):
    """Infer feature transformation based on type
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Append estimator as final step to partial pipeline

        Args:
            X ([pandas.DataFrame]): Features DataFrame
            y ([pandas.Series]): Targets

        Returns:
            [InferedFeaturePipeline]: self
        """
        self.pipeline = infered_pipeline(X)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """Apply fitted pipeline

        Args:
            X ([pandas.DataFrame]): Features DataFrame

        Returns:
            [numpy.array]: Array (of probabilities in case of classifier)
        """
        return self.pipeline.transform(X)
