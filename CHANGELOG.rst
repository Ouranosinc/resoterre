=========
Changelog
=========

`Unreleased <https://github.com/Ouranosinc/resoterre>`_ (latest)
----------------------------------------------------------------

Contributors: Blaise Gauvin St-Denis (:user:`bstdenis`), Trevor James Smith (:user:`Zeitsperre`), Nazim Azeli (:user:`Nazim-crim`), Francis Charette-Migneault (:user:`fmigneault`)

Changes
^^^^^^^
* Drop support for Python 3.10. (:pull:`76`).
* Add support for Python 3.14. (:pull:`65`).
* Add ``data_info`` module. (:pull:`52`).
* Add ``data_io`` module. (:pull:`52`).
* Add ``forecast_utils`` module. (:pull:`52`).
* Add ``geo_utils`` module. (:pull:`52`).
* Add ``hrdps_integrity_check`` module. (:pull:`52`).
* Add ``hrdps_processing`` module. (:pull:`52`).
* Add ``rdps_integrity_check`` module. (:pull:`52`).
* Add ``rdps_processing`` module. (:pull:`52`).
* Add ``rdps_to_hrdps_inference`` module. (:pull:`52`)
* Add ``rdps_to_hrdps_training`` module. (:pull:`52`).
* Add ``nd_plots`` module. (:pull:`52`).
* Add ``calendar_utils`` module. (:pull:`52`).
* Add ``type_checker_utils`` module. (:pull:`52`).
* Add snakemake workflow for RDPS to HRDPS downscaling task. (:pull:`52`).
* Add ``netcdf_utils`` module. (:pull:`47`).
* Add ``timeseries`` module. (:pull:`47`).
* Add ``rdps_to_hrdps`` module. (:pull:`47`).
* Add ``DatasetWithSplits``, ``DatasetWithSave`` and ``DatasetFromNetCDFSave`` classes to ``data_loader_utils`` module. (:pull:`47`).
* Add ``dataset_manager`` module. (:pull:`47`).
* Add ``ml_loops`` module. (:pull:`47`).
* Add ``NeuralNetworksManager`` class to ``network_manager`` module. (:pull:`47`).
* Add ``LinearReLU`` class to ``neural_networks_basic`` module. (:pull:`47`).
* Add ``ml_sample_plot`` module. (:pull:`47`).
* Add ``ml_training_plot`` module. (:pull:`47`).
* Add ``logging_utils`` module. (:pull:`47`).
* Add ``memory_utils`` module. (:pull:`47`).
* Add ``utils`` module. (:pull:`47`).
* Add ``override_config_paths`` function to ``io_utils.py`` to allow overriding config values at runtime. (:pull:`47`).
* Update ``downscaling_inference_rdps_to_hrdps.py`` CLI to support the following arguments for overriding config values. (:pull:`47`).
    * ``--preprocess_batch``: Path to the preprocessed batch file (overrides ``path_preprocessed_batch`` in config).
    * ``--path_models``: Path to the directory containing the pre-trained model (overrides ``path_models`` in config).
    * ``--path_output``: Path to the directory where inference results will be saved (overrides ``path_output`` in config).
* Add ``examples/deploy`` folder with example files for model deployment using Common Workflow Language and Weaver. (:pull:`47`).
    * Add ``unet.cwl`` Common Workflow Language file for model inference.
    * Add ``execute_unet_cwl_schema.yml`` for deployment using weaver.
* Add ``docker`` folder with Dockerfile for resoterre and model inference. (:pull:`47`).
    * Add ``Dockerfile.base`` for resoterre package
    * Add ``Dockerfile.inference`` for model inference using preprocess data.
* Add ``model/`` entry to ``.gitignore`` to avoid model checkpoint inclusion from ``docker/README.md`` procedure.
* Add notebooks/unetToMLM.ipynb to describe the UNet downscaling model by generating a STAC Item and Collection validated with the STAC MLM and Datacube extensions. (:pull:`21`).
* Add UNet option to go to a linear layer at the bottom for 1D inputs. (:pull:`23`).
* Add UNet option to use inputs in the last layer for static features. (:pull:`23`).
* Add ``config_utils`` module. (:pull:`23`).
* Add ``variables`` module. (:pull:`23`).
* Add ``hrdps_variables`` module. (:pull:`23`).
* Add ``rdps_variables`` module. (:pull:`23`).
* Add ``rdps_to_hrdps_workflow`` module. (:pull:`23`).
* Add ``io_utils`` module. (:pull:`23`).
* Add ``runner_unet`` module. (:pull:`23`).
* Add ``snakemake_utils`` module. (:pull:`23`).

Fixes
^^^^^
* Fix typing issues with data_loader_kwargs. (:pull:`64`).
* Fix broken types-PyYAML dependency in environment-dev.yml. (:pull:`64`).
* Set GitHub workflows to support Python version 3.13. (:pull:`22`).
* Set GitHub workflows to support Python version 3.14. (:pull:`65`).

Internal changes
^^^^^^^^^^^^^^^^
* Updated the cookiecutter template. (:pull:`24`):
    * Replaced ``tox.ini`` with new ``tox.toml`` spec
    * Enabled the labelling workflow
    * Updated ``pyproject.toml`` to use PEP 639
    * Added a ``CITATION.cff`` file
    * Replaced the `python-coveralls` dependency (abandoned) for the `coverallsapp/github-action`
    * Updated the ``CODE_OF_CONDUCT.md`` to Contributor Covenant v3.0
* Updated the cookiecutter template. (:pull:`65`):
    * Updated GitHub Workflows to use more secure actions/connections
    * Replaced `pre-commit` with `prek`. Updates pre-commit hooks
    * GenAI contributions guidelines have been added to project files
    * `dependency-groups` are now used for managing development/testing/documentation dependencies (PEP 735)
    * General development/documentation dependency updates.
    * Makefile recipes now handle some dependency management commands
    * `tox.toml` now uses `dependency-groups` and Makefile
    * `pytest` has been updated to v9.0. Now uses TOML-based configuration spec
    * Linting tools now include `vulture`, `deptry`, `yamllint`, and `codespell`
* Updated the cookiecutter template. (:pull:`76`):
    * Restrict token permissions in some overly-permissive workflows.
    * Replace the PyPI source for `types-pyyaml` with the conda package equibalent in `environment.yml`.
    * Update documentation and AI-related guides based on internal discussions.

.. _changes_0.1.2:

`v0.1.2 <https://github.com/Ouranosinc/resoterre/tree/v0.1.2>`_ (2025-08-14)
----------------------------------------------------------------------------

Contributors: Blaise Gauvin St-Denis (:user:`bstdenis`)

Changes
^^^^^^^
* Add ``DenseUNet`` class to ``neural_networks_unet`` module. (:pull:`11`)
* Add ``DenseUNetConfig`` class to ``neural_networks_unet`` module. (:pull:`11`)
* Refactor handling of initialization functions in neural network modules. (:pull:`11`)
* Add ``data_loader_utils`` module. (:pull:`11`)

.. _changes_0.1.1:

`v0.1.1 <https://github.com/Ouranosinc/resoterre/tree/v0.1.1>`_ (2025-07-29)
----------------------------------------------------------------------------

Contributors: Blaise Gauvin St-Denis (:user:`bstdenis`), Trevor James Smith (:user:`Zeitsperre`).

Changes
^^^^^^^
* Add ``network_manager`` module. (:pull:`8`).
    * ``nb_of_parameters`` function to count the number of parameters in a network.
* Add ``neural_networks_basic`` module. (:pull:`8`).
    * ``ModuleWithInitTracker`` and ``ModuleInitFnTracker`` classes to track module initialization functions.
    * ``SEBlock`` class for Squeeze-and-Excitation blocks.
* Add ``neural_networks_unet`` module. (:pull:`8`).
    * ``UNet`` class for U-Net architecture.
* First release of `resoterre` on PyPI.
