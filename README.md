# (Another) Machine Learning Experiment and Operations Pipelines (MLxOPS)

Pipeline for full Machine Learning development lifecycle with metadata.

Model file structure:
```bash
├── mlxops
│   ├── components
│   │   ├── base.py
│   │   ├── data_components.py
│   │   ├── infered_feature_pipeline.py
│   │   ├── __init__.py
│   │   ├── model_components.py
│   │   └── serving_components.py
│   ├── pipeline
│   │   ├── base.py
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   ├── __init__.py
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── requirements.txt
├── setup.py
```

## To install package:
```bash
pip install mlxops
```

# Example 1: Using the different pipeline components

The mlxops components module contains Machine Learning life cycle components of each step. The components consists of:
* DataLoader
* DataFeatureMapper
* DataValidator
* ModelTrainer
* ModelEvaluator
* ModelScore

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# local imports
from mlxops.components import DataLoader, DataFeatureMapper, DataValidator
from mlxops.components import ModelTrainer, ModelEvaluator, ArtifactPusher
from sklearn.model_selection import ShuffleSplit
```

## DataLoader component of the life cycle
Load data using the DataLoader to load data:

```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
data_loader = DataLoader.from_file(file_url, target='target', splitter=ShuffleSplit(n_splits=1, test_size=0.1))
data_loader.run()
```
DataLoader splits data into train, test and evaulation sets. These sets can retrieved from the following properties
```python
data_loader.train_set
data_loader.test_set
data_loader.eval_set
```

## Feature mapper component of the life cycle
This component does for example Normalization, OneHotEncoding etc. If the dtypes are properly set user can use the infered pipeline. For this example we will use the infered feature mapper. The feature mapper's run method takes in a DataLoader object.
```python
feature_mapper = DataFeatureMapper.from_infered_pipeline()
feature_mapper.run(data_loader=data_loader)
```
Example to apply the feature mapper:
```python
mapped_train_features = feature_mapper.transform(data_loader.train_set[0])
```

## Validator component of the life cycle
This component step is optional and it takes in a sklearn validator or outlier detector as input. This validator needs to implement a predict method that returns 1 for inlier and -1 for outlier. The validator component gets applied on feature mapped data from the feature_mapper step. The run method takes in a runned DataLoader and DataValidator as input. 

```python
data_validator = DataValidator(
    validator=IsolationForest(contamination=0.001)
)
data_validator.run(data_loader=data_loader, feature_mapper=feature_mapper)
```
The mask for the train data can be retrieved from a property method. This mask can be used in conjunction with the DataLoader component to select only relevant samples. For example
```python
eval_data, eval_targets = data_loader.train_set
valid_eval_data, valid_eval_targets = eval_data[data_validator.trainset_valid],\
                                      eval_targets[data_validator.trainset_valid]
```

## ModelTrainer Component of the Machine Learning experiment life cycle
The ModelTrainer takes in a sklearn type estimator that implements fit and predict. The ModelTrainer's run method takes components DataLoader, DataFeatureMapper and DataValidator component as inputs. We will set this as our base model.

```python
base_trainer = ModelTrainer(
    estimator=LogisticRegression()
)
base_trainer.run(data_loader, feature_mapper, data_validator)
```
Let save this base model and build a new model. We will compare the new model against the base model later.
```python
base_trainer.save('lr_base_model')
```

Let build a second challenger model using a random forest classifier
```python
new_trainer = ModelTrainer(
    estimator=RandomForestClassifier(n_estimators=50)
)
new_trainer.run(data_loader, feature_mapper, data_validator)
```

## ModelEvaluator component step in the Machine Learning life cycle
The ModelEvaluator component compares two models based on the supplied metrics. he ModelEvaluator's run method takes components DataLoader, DataFeatureMapper and DataValidator component as inputs as well as a new trained model. The base model is an instance atrributes of the class.

```python
evaluator = ModelEvaluator(base_model="base_model",
                           metrics=[accuracy_score])
evaluator.run(
    data_loader, feature_mapper, data_validator, new_trainer
)
```
To check if this model should be pushed:
```python
evaluator.push_model
```
Check the models compare with the supplied metrics:
```python
evaluator.evaluation_metrics
```

# Pipeline example
For a more high level interface. We can build a pipeline. Here we demonstrate a ModelTrainingPipeline. Similar to the example above.

```python
from mlxops.pipeline import ModelTrainingPipeline
```

Setup Pipeline
```python

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"

train_pipeline_arguments = {
    'data_loader': DataLoader.from_file(
        file_name=file_url, target='target', splitter=ShuffleSplit(n_splits=1, test_size=0.25)
    ),
    'data_validator': DataValidator(
        validator=IsolationForest(contamination=0.01)
    ),
    'feature_mapper': DataFeatureMapper.from_infered_pipeline(),
    'trainer': ModelTrainer(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=3)
    ),
    'evaluator': ModelEvaluator(
        base_model='base_model',
        metrics=[accuracy_score]
    ),
    'pusher': ArtifactPusher(model_serving_dir='testfolder'),
    "run_id": "0.0.0.1"
}
```
Execute Pipeline:
```python
train_pipeline = ModelTrainingPipeline(
    **train_pipeline_arguments
)
train_pipeline.run()
train_pipeline.save_model_artifacts("base_model")
```

The ```save_model_artifacts``` saves all component artifacts used in the pipeline along with other metadata. Lets load the saved pipeline
```python
del train_pipeline

loaded_pipeline = ModelTrainingPipeline()
loaded_pipeline.load("base_model")
```
Each component can be access through ```loaded_pipeline.component``` where component can be any of:
```
* data_loader
* data_validator
* evaluator
* feature_mapper
* trainer
* pusher
```
