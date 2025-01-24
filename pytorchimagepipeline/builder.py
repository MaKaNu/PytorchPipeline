import importlib
import os
from pathlib import Path
from typing import Any, Optional

import tomllib

from pytorchimagepipeline.abstractions import Permanence, PipelineProcess
from pytorchimagepipeline.errors import (
    ConfigInvalidTomlError,
    ConfigNotFoundError,
    ConfigPermissionError,
    ConfigSectionError,
    InstTypeError,
    RegistryError,
    RegistryParamError,
)
from pytorchimagepipeline.observer import Observer


class PipelineBuilder:
    def __init__(self):
        self._config: dict[str, Any] = {}
        self._class_registry: dict[str, type] = {}

    def register_class(self, name: str, cls: type) -> Optional[Exception]:
        """Register a class for config-driven instantiation"""
        if not issubclass(cls, (Permanence, PipelineProcess)):
            return RegistryError(name)
        self._class_registry[name] = cls

    def load_config(self, config_path: Path) -> Optional[Exception]:
        """Load and validate configuration from TOML file"""
        config_path_extended = Path("configs") / config_path
        if not config_path_extended.exists():
            return ConfigNotFoundError(config_path_extended)
        if not os.access(config_path_extended, os.R_OK):
            return ConfigPermissionError(config_path_extended)
        with open(config_path_extended, "rb") as f:
            try:
                self._config = tomllib.load(f)
            except tomllib.TOMLDecodeError:
                return ConfigInvalidTomlError(config_path_extended)
        error = self._validate_config_sections()
        return error

    def _validate_config_sections(self) -> Optional[Exception]:
        """Validate required configuration sections"""
        required_sections = ["permanent_objects", "processes"]
        for section in required_sections:
            if section not in self._config:
                return ConfigSectionError(section)

    def build(self) -> tuple[Observer, Optional[Exception]]:
        """Construct the complete pipeline"""
        permanent, error = self._build_permanent_objects()
        if error:
            return None, error
        observer = Observer(permanent_objects=permanent)
        error = self._build_processes(observer)
        if error:
            return None, error
        return observer, None

    def _build_permanent_objects(self) -> tuple[dict[str, Any], Optional[Exception]]:
        """Construct permanent objects with error handling"""
        objects = {}
        for name, config in self._config["permanent_objects"].items():
            objects[name], error = self._instantiate_from_config(name, config)
            if error:
                return {}, error
        return objects, None

    def _build_processes(self, observer: "Observer") -> Optional[Exception]:
        """Build and add processes to observer"""
        for name, config in self._config["processes"].items():
            process, error = self._instantiate_from_config(name, config)
            if error:
                return error
            error = observer.add_process(process)
            if error:
                return error

    def _instantiate_from_config(
        self, context: str, config: dict[str, Any]
    ) -> tuple[Permanence | PipelineProcess, Optional[Exception]]:
        """Safe object instantiation from config"""
        if "type" not in config:
            return None, InstTypeError(context)
        cls_name = config["type"]
        params = config.get("params", {})
        if cls_name not in self._class_registry:
            return None, RegistryError(f"{context}-{cls_name}")
        try:
            return self._class_registry[cls_name](**params), None
        except TypeError:
            return None, RegistryParamError(params)


def get_objects_for_pipeline(pipeline_name: str) -> dict[str, type]:
    full_module_name = "pytorchimagepipeline.pipelines." + pipeline_name
    module = importlib.import_module(full_module_name)
    return module.permanences_to_register | module.processes_to_register


# Usage Example
if __name__ == "__main__":
    objects = get_objects_for_pipeline("sam2segnet")

    builder = PipelineBuilder()
    for key in objects:
        error = builder.register_class(key, objects[key])
        if error:
            raise error

    error = builder.load_config("sam2segnet/execute_pipeline.toml")
    if error:
        raise error
    observer, error = builder.build()
    if error:
        raise error
    observer.run()
