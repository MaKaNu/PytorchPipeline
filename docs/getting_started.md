# Getting Started

> If you come this document to learn how to execute already implemented pipelines, please refer [Usage](usage.md)

This document will teach how to create a new Pipeline, with all necessary parts.

The document is written under the Assumption a developer has already created strong ideas, that should flow into a pipeline.
This could be done on paper, script or notebook, but the final product should already kind of manifested.
Better in text, but in mind is also no issue.

As described in the [Overview](index.md/#overview), the first step is to decide if a part of the Pipeline is a [`Permanence`][{{ permanence }}] or a [`PipelineProcess`][{{ process }}].

## Internal logic

In detail the following image provides some information how the Pipeline is structured.
If you don't want to read the internal logic jump straight to [Create Config](#create-config)

![UML](assets/PytorchImagePipeline.drawio.svg)
///caption
UML Diagram sketch of `PytorchImagePipeline`
///

The [`Observer`][{{ observer }}] is instantiated with a dictionary of zero [`Permanence`][{{ permanence }}] or more.
This step is displayed as the aggregation[^1].
Afterwards one [`PipelineProcesse`][{{ process }}] or more are added to the [`Observer`][{{ observer }}].
This step is displayed as the composition[^1].
The run method of the [`Observer`][{{ observer }}] class iterates over each [`PipelineProcess`][{{ process }}] and calls it [`execute`][{{ process_execute }}] method, which uses (displayed as association[^1]) the observer itself to access the [`Permanence`][{{ permanence }}] if needed.

[^1]: A short explanation between an [aggregation vs. composition vs. association](https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-aggregation-vs-composition/).

Since doing this by hand is cumbersome and error-prone, a [`PipelineBuilder`][{{ builder }}] class is provided, which executes this behavior.

## Create Config

After the decision which Parts fall into either the [`Permanence`][{{ permanence }}] or the [`PipelineProcess`][{{ process }}] category, the creation of the config files begins.

> It is not necessary to complete the configs at this point. The goal is to create a structure to begin with and complete later.

Following steps are necessary to create the config structure:

1. Create new folder under configs with the name of the Pipeline.
2. Create `execute_pipeline.toml` file inside this folder.
3. Create `initial_pipeline.toml` file inside this folder.

### The `initial_pipeline.toml`

Since not all components of a Pipeline should be part of the repository, it is necessary to provide such information.

Following components of a Pipeline needed to be defined inside this config:

- Dependencies
- Dataset locations
- Model locations

An Example config could look like the following:

```toml
[dependencies.submodules.sam]
name = "segment_anything"
url = "https://github.com/facebookresearch/segment-anything.git"
commit = "dca509fe"

[data.datasets.pascal]
location = "/mnt/data1/datasets/image/voc/VOCdevkit/VOC2012/"
format = "pascalvoc"

[data.models.sam]
location = "/mnt/data1/models/sam/sam_vit_h_4b8939.pth"
format = "pth"
```

The dependency is in this case a git submodule, but could be also a pip package.
The dependencies will be installed via `uv pip install`, so it will not interfere with `pyproject.toml`.

A dataset and a model are always provided with a key (`pascal` or `sam` in this case), a location and a format.
The key will be used to create the symlink below `data/dataset` or `data/models` based on location.
The format helps to reduce Boilerplate for dataset creation.

### The `execute_pipeline.toml`

This config file is used to provide the different [`Permanence`][{{ permanence }}] and [`PipelineProcess`][{{ process }}] object configurations.

An example config could look like the following:

```toml
[permanences.data]
type = "Datasets"
params = { root = "localhost", format = "pascalvoc" }

[processes.dummy]
type = "DummyProcess"
```

Here are one [`Permanence`][{{ permanence }}] and one [`PipelineProcess`][{{ process }}] object provided.
The `params` field is always Optional.
The `type` field provides the classes of the Pipeline, which we need to create inside our [Pipeline Library](#create-pipeline-library).
The `params` need to match accordingly to the `__init__` method of the Implementation.

Additional configuration for provided [`Permanence`][{{ permanence }}] or [`PipelineProcess`][{{ process }}] classes could here be provided.
As example from the core package the `Visualization` process is defined in the config.
The [`Builder`][{{ builder }}] will first register the `Visualization` class from core package.
It is also possible to override the `Visualization` class inside the pipeline library.

## Create Pipeline Library

With the initiation of the config files we can begin creating the actual pipeline library.
It is necessary to create for every entry in the config its implementation, but we have the freedom to provide additional pipeline internal structure.
The structure of a new pipeline subpackage looks the following:

```tree
📦pytorchimagepipeline
 ┗ 📦pipelines
    ┗ 📦node_modules
       ┣ 📜__init__.py
       ┣ 📜permanences.py
       ┗ 📜processes.py
```

Following the structure the implementations should be provided dependent on the config either inside the `permanences.py` or `processes.py` module.
This is just a recommendation to follow the pattern of the package, but a developer can decide against the pattern.
The only necessary part is the announcement of the [`Permanence`][{{ permanence }}] and the [`PipelineProcess`][{{ process }}] inside the subpackage `__init__.py` module.

For the previous example the `__init__.py` module looks the following:

```python
from permaneces import Datasets
from processes import DummyProcess

permanences_to_register = {"Datasets": Datasets}
processes_to_register = {"DummyProcess": DummyProcess}
```

## Execute or testing

From that point there are initial two possible ways to proceed.

1. Execute the Pipeline directly

This could be a good start point to get familiar how the pipeline works and looks as the final product.
It is good practice to follow this approach if the developer works first time with this project.
Nevertheless, the second way is an absolute recommendation, to create the final pipeline.

2. Creating test cases

The creation of test cases is a recommended approach, since at the end multiple small components of the pipeline are provided.
Instead of running all the time the complete pipeline, checking each bit with artificial and abstract test cases could provide spare time.
Not only this, after iterating through changes in the pipeline library, we are able to verify the correctness of the step in the Pipeline.
In a certain situation it might be necessary to extend the test case to fulfil new challenges, which were not thought about at the beginning of the Pipeline development.
