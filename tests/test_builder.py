from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Literal
from unittest.mock import mock_open, patch

import pytest

try:
    from tomllib import TOMLDecodeError
except ImportError:
    try:
        from tomli import TOMLDecodeError  # type: ignore  # noqa: PGH003
    except ImportError:
        sys.exit("Error: This program requires either tomllib or tomli but neither is available")

from pytorchimagepipeline.abstractions import Permanence, PipelineProcess
from pytorchimagepipeline.builder import PipelineBuilder, get_objects_for_pipeline
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
        print("Running cleanup")
        return None


class MockedPipelineProcess(PipelineProcess):
    def execute(self, observer):
        print(f"Running execute with {observer}")
        return None


class TestPipelineBuilder:
    pipeline_builder: PipelineBuilder

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
    def test_register_class(self, name: str, cls: type, expected_error: None | type[Exception]):
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
            ({"permanences": {}, "processes": {}}, None),
            ({"permanences": {}}, "processes"),
            ({"processes": {}}, "permanences"),
            ({}, "permanences"),
        ],
        ids=("Valid", "MissingProcess", "MissingPermanent", "MissingBoth"),
    )
    def test_validate_config_sections(
        self,
        input_config: dict[str, dict[Any, Any]],
        expected_error_section: None | Literal["processes"] | Literal["permanences"],
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
            ("valid_file.toml", True, True, '{"permanences": {}, "processes": {}}', None),
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
                "pytorchimagepipeline.builder.toml_load",
                side_effect=TOMLDecodeError if file_content == "invalid_toml" else lambda f: ast.literal_eval(f.read()),
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
        "permanences_config, expected_error",
        [
            ({"object1": {"type": "MockedPermanence"}}, None),  # Valid object
            ({"object1": {"type": "UnknownClass"}}, RegistryError),  # Unknown class
            ({"object1": {"type": "MockedPermanence", "params": {"invalid": 1}}}, RegistryParamError),  # Invalid params
        ],
        ids=("ValidObjects", "InvalidClass", "InvalidParams"),
    )
    def test_build_permanences(self, permanences_config, expected_error):
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder._config["permanences"] = permanences_config

        objects, error = self.pipeline_builder._build_permanences()

        if expected_error is None:
            assert error is None
            assert len(objects) == len(permanences_config)
        else:
            assert isinstance(error, expected_error)
            assert str(error)
            assert objects == {}

    @pytest.mark.parametrize(
        "config, expected_error",
        [
            (
                {
                    "permanences": {"object1": {"type": "MockedPermanence"}},
                    "processes": {"process1": {"type": "MockedPipelineProcess"}},
                },
                None,
            ),  # Valid case
            (
                {
                    "permanences": {"object1": {"type": "UnknownClass"}},
                    "processes": {"process1": {"type": "PipelineProcess"}},
                },
                RegistryError,
            ),  # Invalid permanent object
            (
                {
                    "permanences": {"object1": {"type": "MockedPermanence"}},
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
            assert len(observer._permanences) == len(config["permanences"])
            assert len(observer._processes) == len(config["processes"])
        else:
            assert isinstance(error, expected_error)

    @pytest.mark.parametrize(
        "pipeline_name, module_exists, expected_error",
        [
            ("valid_pipeline", True, None),
            ("invalid_pipeline", False, ModuleNotFoundError),
        ],
        ids=("ValidPipeline", "InvalidPipeline"),
    )
    def test_get_objects_for_pipeline(self, pipeline_name, module_exists, expected_error):
        mock_module = type(
            "MockModule",
            (),
            {
                "permanences_to_register": {"perm1": MockedPermanence},
                "processes_to_register": {"proc1": MockedPipelineProcess},
            },
        )

        with patch("importlib.import_module") as mock_import:
            if module_exists:
                mock_import.return_value = mock_module
            else:
                mock_import.side_effect = ModuleNotFoundError()

            objects, error = get_objects_for_pipeline(pipeline_name)

            if expected_error is None:
                assert error is None
                assert len(objects) == 2
                assert "perm1" in objects
                assert "proc1" in objects
                assert objects["perm1"] is MockedPermanence
                assert objects["proc1"] is MockedPipelineProcess
            else:
                assert isinstance(error, expected_error)
                assert objects == {}
