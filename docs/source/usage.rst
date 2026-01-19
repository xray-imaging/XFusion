=====
Usage
=====

This page summarizes the main workflow and CLI entry points.

Overview
========

Currently, XFusion supports two model families for high-quality x-ray image sequence reconstruction:

* EDVR
* Swin vision transformer (SwinIR-style)

Package installation
====================

From the project root directory::

    pip install .

Initialization
==============

Initialize a new configuration file::

    xfusion init --model_type [EDVRModel or SwinIRModel]

After initialization, a configuration file named ``xfusion.conf`` will be generated in your home directory.
This configuration file is updated automatically during XFusion workflows.

Data preparation
================

Data for model pre-training
---------------------------

Download the REDS datasets:

* Sharp dataset: ``train_sharp``
* Low-resolution dataset: ``train_sharp_bicubic``

See the `REDS dataset page <https://seungjunnah.github.io/Datasets/reds>`_ and set the corresponding directories
in the ``convert`` section of your configuration file.

Data for fine-tuning
--------------------

Fine-tuning data are not available at this moment.

Data for testing
----------------

Two sample test datasets can be downloaded from `Tomobank <https://tomobank.readthedocs.io/en/latest/source/data/docs.data.radio.html>`_.

Data conversion
===============

Convert REDS data to grayscale::

    xfusion convert \
      --dir-lo-convert [directory/to/low_resolution/RGB/training_images] \
      --dir-hi-convert [directory/to/high_resolution/RGB/training_images] \
      --out-dir-lo [directory/to/low_resolution/grayscale/training_images] \
      --out-dir-hi [directory/to/high_resolution/grayscale/training_images]

Training
========

Run training::

    xfusion train \
      --dir-lo-train [directory/to/low_resolution/grayscale/training_images] \
      --dir-hi-train [directory/to/high_resolution/grayscale/training_images] \
      --dir-lo-val [directory/to/low_resolution/grayscale/validation_images] \
      --dir-hi-val [directory/to/high_resolution/grayscale/validation_images] \
      --opt [directory/to/training_settings.yaml] \
      --path-train-meta-info-file [directory/to/train_meta_info_file] \
      --path-val-meta-info-file [directory/to/val_meta_info_file] \
      --pretrain_network_g [path/to/pretrained_model_weights]

Test data download
==================

Download test data from Tomobank::

    xfusion download \
      --dir-inf [tomobank/link/address/of_test_dataset] \
      --out-dir-inf [directory/to/testing_images]

Inference
=========

Run inference::

    xfusion inference \
      --opt [directory/to/testing_dataset_settings.yaml] \
      --arch-opt [directory/to/training_settings.yaml] \
      --model_file [path/to/model_weights] \
      --machine [tomo or polaris]

Notes
-----

* EDVRModel currently runs in single-process mode.
* SwinIRModel currently runs in multi-process mode.
