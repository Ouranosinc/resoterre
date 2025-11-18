=========
Changelog
=========

..
    `Unreleased <https://github.com/Ouranosinc/resoterre>`_ (latest)
    ----------------------------------------------------------------

    Contributors: Blaise Gauvin St-Denis (:user:`bstdenis`)

    Changes
    ^^^^^^^
    * No change.

    Fixes
    ^^^^^
    * Set github workflows python version to 3.13.

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
