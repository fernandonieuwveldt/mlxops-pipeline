"""MLxOPS persistance interface"""

from abc import ABC, abstractmethod
import pathlib
import json
import pickle

from mlxops.components import _COMPONENT_MAPPER


# TODO: create one object(container) that holds all components???


class _BasePersistor(ABC):
    """Base interface for saving and loaded component and pipeline artifacts. All methods
    are classmethods
    """
    @classmethod
    def save_component(cls, artifact, artifact_dir):
        pickle.dump(
            artifact, open(f"{artifact_dir}/{artifact.__class__.__name__}.pkl", 'wb')
        )

    @classmethod
    def save(cls, artifact, artifact_dir):
        """Save metadata to disk

        Args:
            component: component to be saved
            artifact_dir ([str]): location of artifact
        """
        artifact_path = pathlib.Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        for instance_name, component in artifact.__dict__.items():
            if hasattr(component, 'run'):
                cls.save_component(component, artifact_dir)

    @abstractmethod
    def load(self, artifact_dir):
        pass


class PersistComponent(_BasePersistor):
    """Saving and loading for pipeline components
    """
    @classmethod
    def load(cls, artifact_dir):
        """Load saved metadata

        Args:
            artifact_dir ([str]): location of artifact

        Returns:
            [type]: [description]
        """
        with open(f"{artifact_dir}", 'rb') as pfile:
            artifact = pickle.load(pfile)
        return artifact


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
        super().save(pipeline, artifact_dir)

        component_instance_connector = {}
        for instance_name, component in pipeline.__dict__.items():
            if hasattr(component, 'run'):
                component_instance_connector[component.__class__.__name__] = instance_name

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


def load_component(artifact_dir):
    """load the component artifact

    Args:
        artifact_dir (str): directory location

    Returns:
        component: loaded component
    """
    loaded_component = PersistComponent.load(artifact_dir) 
    return loaded_component


def save_component(component, artifact_dir):
    """Save component artifacts

    Args:
        component: component to be saved
        artifact_dir (str): directory location
    """
    PersistComponent.save(component, artifact_dir)
    return


def save_pipeline(component, artifact_dir):
    """Save component artifacts

    Args:
        component: component to be saved
        artifact_dir (str): directory location
    """
    PersistPipeline.save(component, artifact_dir)
    return


def load_pipeline(artifact_dir):
    """load all saved artifacts from pipeline run

    Args:
        artifact_dir (str): directory location

    Returns:
        pipeline: loaded pipeline components
    """
    pipeline = PersistPipeline()
    pipeline.load(artifact_dir)
    return pipeline


def save(obj, artifact_dir):
    """Save component or pipeline artifacts. This method serves as factory for saving
    different object types

    Args:
        obj: obj to be saved
        artifact_dir (str): directory location
    """
    if obj._type == "component":
        PersistComponent.save(obj, artifact_dir)
        return
    if obj._type == "pipeline":
        PersistPipeline.save(obj, artifact_dir)
        return
    raise TypeError("object to be pickled not of type component or pipeline")


def load(artifact_dir):
    """load component or pipeline artifacts. This method serves as factory for loading
    different object types

    Args:
        artifact_dir (str): directory location

    Returns:
        [component, pipeline]: loaded component or pipeline
    """
    artifact_path = pathlib.Path(artifact_dir)
    if not artifact_path.exists():
        raise FileNotFoundError(f"No such file or directory: {artifact_dir}")
    if artifact_path.is_file() and artifact_path.suffix == '.pkl':
        return load_component(artifact_dir)
    if artifact_path.is_dir():
        return load_pipeline(artifact_dir)
