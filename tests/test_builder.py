import ast
from pathlib import Path
from typing import Any, Literal, Optional
from unittest.mock import mock_open, patch

import pytest
import tomllib

from pytorchimagepipeline.abstractions import Permanence, PipelineProcess
from pytorchimagepipeline.builder import PipelineBuilder
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


class MockedPermanence(Permanence):
    def cleanup(self):
        return super().cleanup()


class MockedPipelineProcess(PipelineProcess):
    def execute(self, observer):
        return super().execute(observer)


class TestPipelineBuilder:
    @pytest.fixture(autouse=True, scope="class")
    def fixture_class(self):
        cls = type(self)
        cls.pipeline_builder = PipelineBuilder()
        yield

    @pytest.mark.parametrize(
        "name, cls, expected_error",
        [
            ("ValidClass1", Permanence, None),  # Valid registration
            ("ValidClass2", PipelineProcess, None),  # Valid registration
            ("InvalidClass1", dict, RegistryError),  # Invalid class
            ("InvalidClass2", str, RegistryError),  # Invalid class
        ],
        ids=("ValidPermance", "ValidPipelineProcess", "InvalidDict", "InvalidStr"),
    )
    def test_register_class(self, name: str, cls: type, expected_error: Optional[Exception]):
        error = self.pipeline_builder.register_class(name, cls)

        if expected_error is None:
            assert error is None
            assert name in self.pipeline_builder._class_registry
            assert self.pipeline_builder._class_registry[name] is cls
        else:
            assert isinstance(error, expected_error)
            assert str(error)
            assert name not in self.pipeline_builder._class_registry

    @pytest.mark.parametrize(
        "input_config, expected_error_section",
        [
            #
            ({"permanent_objects": {}, "processes": {}}, None),
            ({"permanent_objects": {}}, "processes"),
            ({"processes": {}}, "permanent_objects"),
            ({}, "permanent_objects"),
        ],
        ids=("Valid", "MissingProcess", "MissingPermanent", "MissingBoth"),
    )
    def test_validate_config_sections(
        self,
        input_config: dict[str, dict[Any, Any]],
        expected_error_section: None | Literal["processes"] | Literal["permanent_objects"],
    ):
        self.pipeline_builder._config = input_config
        error = self.pipeline_builder._validate_config_sections()

        if expected_error_section is None:
            assert not error
        else:
            assert isinstance(error, ConfigSectionError)
            assert str(error)

    @pytest.mark.parametrize(
        "config_path, file_exists, readable, file_content, expected_error",
        [
            ("missing_file.toml", False, False, None, ConfigNotFoundError),
            ("unreadable_file.toml", True, False, None, ConfigPermissionError),
            ("invalid_file.toml", True, True, "invalid_toml", ConfigInvalidTomlError),
            ("missing_section.toml", True, True, '{"unrelated_key": "value"}', ConfigSectionError),
            ("valid_file.toml", True, True, '{"permanent_objects": {}, "processes": {}}', None),
        ],
        ids=("MissingConfig", "UnreadableConfig", "InvalidTOML", "MissingSection", "ValidConfig"),
    )
    def test_load_config(self, config_path, file_exists, readable, file_content, expected_error):
        # Mocking file system behavior
        with (
            patch("pathlib.Path.exists", return_value=file_exists),
            patch("os.access", return_value=readable),
            patch("builtins.open", mock_open(read_data=file_content) if file_content else None),
            patch(
                "tomllib.load",
                side_effect=tomllib.TOMLDecodeError
                if file_content == "invalid_toml"
                else lambda f: ast.literal_eval(f.read()),
            ),
        ):
            error = self.pipeline_builder.load_config(Path(config_path))

            if expected_error is None:
                assert error is None
                assert self.pipeline_builder._config is not None
            else:
                assert isinstance(error, expected_error)
                assert str(error)

    @pytest.mark.parametrize(
        "context, config, expected_error",
        [
            ("context1", {}, InstTypeError),
            ("context2", {"type": "UnknownClass"}, RegistryError),
            ("context3", {"type": "MockedPermanence", "params": {"invalid": 1}}, RegistryParamError),
            ("context4", {"type": "MockedPermanence"}, None),
            ("context5", {"type": "MockedPipelineProcess"}, None),
        ],
        ids=("MissingType", "InvalidClass", "InvalidParams", "ValidPerma", "ValidProcess"),
    )
    def test_instantiate_from_config(self, context, config, expected_error):
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder.register_class("MockedPipelineProcess", MockedPipelineProcess)
        result, error = self.pipeline_builder._instantiate_from_config(context, config)

        if expected_error is None:
            assert error is None
            assert result is not None
        else:
            assert isinstance(error, expected_error)
            assert str(error)
            assert result is None

    @pytest.mark.parametrize(
        "processes_config, expected_error",
        [
            ({"process1": {"type": "MockedPipelineProcess"}}, None),
            ({"process1": {"type": "UnknownClass"}}, RegistryError),
            ({"process1": {"type": "MockedPipelineProcess", "params": {"invalid": 1}}}, RegistryParamError),
            ({"process1": {"type": "MockedPermanence"}}, InstTypeError),
        ],
        ids=("ValidProcess", "InvalidProcess", "InvalidParams", "InvalidPermanence"),
    )
    def test_build_processes(self, processes_config, expected_error):
        self.pipeline_builder.register_class("MockedPipelineProcess", MockedPipelineProcess)
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder._config["processes"] = processes_config
        observer = Observer({})

        error = self.pipeline_builder._build_processes(observer)

        if expected_error is None:
            assert error is None
            assert len(observer._processes) == len(processes_config)
        else:
            assert isinstance(error, expected_error)
            assert str(error)

    @pytest.mark.parametrize(
        "permanent_objects_config, expected_error",
        [
            ({"object1": {"type": "MockedPermanence"}}, None),  # Valid object
            ({"object1": {"type": "UnknownClass"}}, RegistryError),  # Unknown class
            ({"object1": {"type": "MockedPermanence", "params": {"invalid": 1}}}, RegistryParamError),  # Invalid params
        ],
        ids=("ValidObjects", "InvalidClass", "InvalidParams"),
    )
    def test_build_permanent_objects(self, permanent_objects_config, expected_error):
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder._config["permanent_objects"] = permanent_objects_config

        objects, error = self.pipeline_builder._build_permanent_objects()

        if expected_error is None:
            assert error is None
            assert len(objects) == len(permanent_objects_config)
        else:
            assert isinstance(error, expected_error)
            assert str(error)
            assert objects == {}

    @pytest.mark.parametrize(
        "config, expected_error",
        [
            (
                {
                    "permanent_objects": {"object1": {"type": "MockedPermanence"}},
                    "processes": {"process1": {"type": "MockedPipelineProcess"}},
                },
                None,
            ),  # Valid case
            (
                {
                    "permanent_objects": {"object1": {"type": "UnknownClass"}},
                    "processes": {"process1": {"type": "PipelineProcess"}},
                },
                RegistryError,
            ),  # Invalid permanent object
            (
                {
                    "permanent_objects": {"object1": {"type": "MockedPermanence"}},
                    "processes": {"process1": {"type": "UnknownClass"}},
                },
                RegistryError,
            ),  # Invalid process
        ],
        ids=("ValidBuild", "InvalidPerma", "InvalidProcess"),
    )
    def test_build(self, config, expected_error):
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder.register_class("MockedPipelineProcess", MockedPipelineProcess)
        self.pipeline_builder._config = config

        observer, error = self.pipeline_builder.build()

        if expected_error is None:
            assert error is None
            assert observer is not None
            assert len(observer._permanent_objects) == len(config["permanent_objects"])
            assert len(observer._processes) == len(config["processes"])
        else:
            assert isinstance(error, expected_error)
            assert observer is None
