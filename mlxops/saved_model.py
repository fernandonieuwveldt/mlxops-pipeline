"""MLxOPS persistance interface"""

from abc import ABC, abstractmethod
import pathlib
import json
import pickle

from mlxops.components import _COMPONENT_MAPPER


class _BasePersistor(ABC):
    """Base interface for saving and loaded component and pipeline artifacts. All methods
    are classmethods
    """
    @abstractmethod
    def save(self, artifact, artifact_dir):
        pass

    @abstractmethod
    def load(self, artifact_dir):
        pass


class PersistComponent(_BasePersistor):
    """Saving and loading for pipeline components
    """
    @classmethod
    def save(cls, artifact, artifact_dir):
        """Save metadata to disk

        Args:
            component: component to be saved
            artifact_dir ([str]): location of artifact
        """
        pickle.dump(
            artifact, open(f"{artifact_dir}/{artifact.__class__.__name__}.pkl", 'wb')
        )

    @classmethod
    def load(cls, artifact_dir):
        """Load saved metadata

        Args:
            artifact_dir ([str]): location of artifact

        Returns:
            [type]: [description]
        """
        return pickle.load(
            open(f"{artifact_dir}", 'rb')
        )


class PersistPipeline(_BasePersistor):
    """Saving and loading of full pipelines. All classes that has run method will be saved
    and loaded
    """
    @classmethod
    def save(cls, pipeline, artifact_dir=None):
        """Save all training artifacts set at training

        Args:
            artifact_dir ([str]): Folder location to save artifacts
        """
        artifact_path = pathlib.Path(artifact_dir)
        artifact_path.mkdir(parents=True)
        component_instance_connector = {}
        for instance_name, component in pipeline.__dict__.items():
            if hasattr(component, 'run'):
                component_instance_connector[component.__class__.__name__] = instance_name
                save_component(component, artifact_dir)

        with open(f"{artifact_dir}/component_instance_connector.json", 'w') as jsonfile:
            json.dump(component_instance_connector, jsonfile)

        with open(f"{artifact_dir}/run_uuid.json", 'w') as jsonfile:
            json.dump(pipeline.run_id, jsonfile)

        # with open(f"{folder_name}/component_metadata.json", 'w') as jsonfile:
        #     json.dump(self.model_training_metadata, jsonfile)

    def load(self, artifact_dir=None):
        """Load final state of model training Pipeline.

        Args:
            folder_name ([str]): Folder name containing saved artifacts
        """
        with open(f"{artifact_dir}/run_uuid.json", 'r') as jsonfile:
            run_uuid = json.load(jsonfile)

        setattr(self, "run_id", run_uuid)

        with open(f"{artifact_dir}/component_instance_connector.json", 'r') as jsonfile:
            component_instance_connector = json.load(jsonfile)

        for name, component in component_instance_connector.items():
            component = _COMPONENT_MAPPER[name]
            component_attr_name = component_instance_connector[name]
            setattr(
                self, component_attr_name, load_component(f"{artifact_dir}/{name}.pkl")
            )


class PersistorFactory:
    """Concrete class to select
    """
    def __init__(self):
        pass

    def save(self, component, artifact_dir):
        """Save component artifacts

        Args:
            component: component to be saved
            artifact_dir (str): directory location
        """

    def load(self, artifact_dir):
        """load the component artifact

        Args:
            artifact_dir (str): directory location

        Returns:
            component: loaded component
        """


def load_component(artifact_dir):
    """load the component artifact

    Args:
        artifact_dir (str): directory location

    Returns:
        component: loaded component
    """
    _loaded_component = PersistComponent.load(artifact_dir) 
    return _loaded_component


def save_component(component, artifact_dir):
    """Save component artifacts

    Args:
        component: component to be saved
        artifact_dir (str): directory location
    """
    PersistComponent.save(component, artifact_dir)
    return


def load(artifact_dir):
    """load the component artifact

    Args:
        artifact_dir (str): directory location

    Returns:
        component: loaded component
    """
    _pipeline = PersistPipeline()
    _pipeline.load(artifact_dir)
    return _pipeline


def save(pipeline, artifact_dir):
    """Save component artifacts

    Args:
        component: component to be saved
        artifact_dir (str): directory location
    """
    PersistPipeline.save(pipeline, artifact_dir)
    return
