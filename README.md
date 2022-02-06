# Automating the ML Training Lifecycle with MLxOPS

We tend to reinvent the training lifecycle on different projects. Given that most ML training lifecycle consist of similar steps
we can automate most of the parts of an ML project. 

This package contains model training life cycle components for automating the training and scoring life cycle of models.
Pandas dataframes are used as the underlying data object. As for the estimators, it can be sklearn type estimators that has 
fit and predict method. Estimators like XGBoost en LightGBM can also be used. 

MLxOPS serves as experiment and operations module by also persisting all metadata for each component and any artifacts we need to 
reproduce results.

The mlxops components module contains Machine Learning life cycle components of each step. The components consists of:
* DataLoader
    * Data loading component of the training pipeline. This component also contains any stateless preprocessing steps the user
      wants to apply on the data.
* DataFeatureMapper
    * Feature processing pipeline. Apply Feature mapper for example Normalization for numeric features and OneHotEncoder for
      categoric features. Mapper here is stateful.
* DataValidator
    * Data validation component of the training pipeline. Can use sklearn outlier detectors or use can implement their own. This validator
      should return a mask where -1 is an outlier and 1 an inlier.
* ModelTrainer
    * Model component of the training pipeline. ModelTrainer depends on a runned DataLoader, DataFeatureMapper and DataValidator to fit estimator.
* ModelEvaluator
    * Compare metrics between trained model and current model in production. This can be any of the metrics implemented in sklearn
* ArtifactPusher
    * Pushes artifacts to PROD directory if current model is an improvement on the current best model

<br />

The mlxops package also contains high level interfaces for training and scoring using pipeline modules:
* ModelTrainingPipeline
    * This is a high level implementation of the training life cycle for all the steps in life cycle
* ScoringPipeline
    * This pipeline can be used to score data. The model can be loaded from disk or supplied.

## To install package:
```bash
pip install mlxops
```

# Example 1: Using the different pipeline components


```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

# local imports
import mlxops
from mlxops.components import DataLoader, DataFeatureMapper, DataValidator
from mlxops.components import ModelTrainer, ModelEvaluator, ArtifactPusher
```

## DataLoader component of the life cycle
Load data using the DataLoader to load data:

```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
data_loader = DataLoader.from_file(file_url, target='target', splitter=ShuffleSplit)
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
train_data, train_targets = data_loader.train_set
valid_train_data, valid_train_targets = train_data[data_validator.trainset_valid],\
                                        train_targets[data_validator.trainset_valid]
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
mlxops.saved_model.save(base_trainer, "lr_base_model")
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
evaluator = ModelEvaluator(base_model="lr_base_model",
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
        file_name=file_url, target='target', splitter=ShuffleSplit
    ),
    'data_validator': DataValidator(
        validator=IsolationForest(contamination=0.01)
    ),
    'feature_mapper': DataFeatureMapper.from_infered_pipeline(),
    'trainer': ModelTrainer(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=3)
    ),
    'evaluator': ModelEvaluator(
        base_model='lr_base_model',
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
mlxops.saved_model.save(train_pipeline, "base_model")
```

The ```saved_model``` module saves all component artifacts used in the pipeline along with other metadata. Lets load the saved pipeline
```python
del train_pipeline

loaded_pipeline = mlxops.saved_model.load("base_model")
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
