import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from sklearn_mlops_pipelines.components import DataLoader, DataFeatureMapper, DataInputValidator
from sklearn_mlops_pipelines.components import ModelTrainer, ModelEvaluator

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
target = dataframe.pop('target')

data_loader = DataLoader.from_file(file_url, target=target)
data_loader.run()

feature_mapper = DataFeatureMapper.from_infered_pipeline()
feature_mapper.run(data_loader=data_loader)
feature_mapper.feature_pipeline.transform(data_loader.train_set[0])

data_validator = DataInputValidator(
    validator=IsolationForest(contamination=0.1)
)
data_validator.run(data_loader=data_loader, feature_mapper=feature_mapper)

base_trainer = ModelTrainer(
    estimator=LogisticRegression()
)
base_trainer.run(data_loader=data_loader, feature_mapper=feature_mapper, data_validator=data_validator)

new_trainer = ModelTrainer(
    estimator=RandomForestClassifier(n_estimators=50)
)
new_trainer.run(data_loader=data_loader, feature_mapper=feature_mapper, data_validator=data_validator)

# Evaluator
evaluator = ModelEvaluator(base_model=base_trainer,
                           new_model=new_trainer,
                           metrics=[accuracy_score])
evaluator.run(data_loader=data_loader, feature_mapper=feature_mapper, data_validator=data_validator)
