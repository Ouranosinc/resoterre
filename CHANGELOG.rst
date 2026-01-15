=========
Changelog
=========

`Unreleased <https://github.com/Ouranosinc/resoterre>`_ (latest)
----------------------------------------------------------------

Contributors: Blaise Gauvin St-Denis (:user:`bstdenis`), Trevor James Smith (:user:`Zeitsperre`), Nazim Azeli (:user:`Nazim-crim`).

Changes
^^^^^^^
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
* Set GitHub workflows Python version to 3.13 (:pull:`22`).

Internal changes
^^^^^^^^^^^^^^^^
* Updated the cookiecutter template (:pull:`24`):
    * Replaced ``tox.ini`` with new ``tox.toml`` spec
    * Enabled the labelling workflow
    * Updated ``pyproject.toml`` to use PEP 639
    * Added a ``CITATION.cff`` file
    * Replaced the `python-coveralls` dependency (abandoned) for the `coverallsapp/github-action`
    * Updated the ``CODE_OF_CONDUCT.md`` to Contributor Covenant v3.0

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
