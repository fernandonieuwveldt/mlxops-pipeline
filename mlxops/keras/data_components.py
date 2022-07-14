from base.data_components import BaseDataLoader


N_SPLITS = 1
SPLIT_RATIOS = {
    'train_ratio': 0.75,
    'eval_ratio': 0.15,
    'test_ratio': 0.10
}


class TensorflowDataLoader(BaseDataLoader):

    def __init__(self, data, target=None, feature_transformer=[], split_ratios=None):
        if target is not None:
            self.target = data.pop(target)
        self.data = data
        self.feature_transformer = feature_transformer
        if self.feature_transformer:
            # nothing for now
        if split_ratios is None:
            self.split_ratios = self.SPLIT_RATIOS
        # preprocessors should be stateless
        self._train_set = None
        self._test_set = None
        self._eval_set = None

    