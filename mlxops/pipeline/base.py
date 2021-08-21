"""Base class for all pipelines"""

import pathlib
import json
import pickle

from mlxops.components import _COMPONENT_MAPPER


class BasePipeline:
    """Base class for all Pipeline. Contains metadata registring and saving final component state
    during pipeline"""

    @property
    def model_training_metadata(self):
        """Return metadata of all runned pipeline components
        """
        return {component.__class__.__name__: component.metadata
                for _, component in self.__dict__.items() if hasattr(component, 'run')
        }

    def save_model_artifacts(self, artifact_dir=None):
        """Save all training artifacts set at training

        Args:
            artifact_dir ([str]): Folder location to save artifacts
        """
        artifact_path = pathlib.Path(artifact_dir)
        artifact_path.mkdir(parents=True)

        component_instance_connector = {}
        for instance_name, component in self.__dict__.items():
            if hasattr(component, 'run'):
                component_instance_connector[component.__class__.__name__] = instance_name
                component.save(artifact_dir)

        with open(f"{artifact_dir}/component_instance_connector.json", 'w') as jsonfile:
            json.dump(component_instance_connector, jsonfile)

        with open(f"{artifact_dir}/run_uuid.json", 'w') as jsonfile:
            json.dump(self.run_id, jsonfile)

        # with open(f"{folder_name}/component_metadata.json", 'w') as jsonfile:
        #     json.dump(self.model_training_metadata, jsonfile)

    def load(self, folder_name=None):
        """Load final state of model training Pipeline.

        Args:
            folder_name ([str]): Folder name containing saved artifacts
        """
        artifacts = pathlib.Path(folder_name)

        with open(f"{folder_name}/component_instance_connector.json", 'r') as jsonfile:
            component_instance_connector = json.load(jsonfile)

        with open(f"{folder_name}/run_uuid.json", 'r') as jsonfile:
            run_uuid = json.load(jsonfile)
        setattr(self, "run_id", run_uuid)

        for name, component in component_instance_connector.items():
            component = _COMPONENT_MAPPER[name]
            component_attr_name = component_instance_connector[name]
            setattr(self, component_attr_name, component.load(folder_name))
