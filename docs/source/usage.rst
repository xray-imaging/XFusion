=====
Usage
=====

This page summarizes the main workflow and CLI entry points.

Overview
========

XFusion is a deep learning-based spatiotemporal fusion approach for high-fidelity ultra-high-speed
x-ray radiography. It reconstructs high-quality x-ray images by combining the high spatial resolution
of a high-speed camera with the high temporal resolution of an ultra-high-speed camera image sequence.

Models and methods for spatial and temporal calibration are also included.

This implementation is based on the `BasicSR toolbox <https://github.com/XPixelGroup/BasicSR>`_
and the `RAFT model <https://github.com/princeton-vl/RAFT>`_.

Currently, XFusion supports two model families for high-quality xray image sequence reconstruction:

* EDVR
* Swin vision transformer (SwinIR-style)

In addition, XFusion is functional with x-ray image sequences from both:

* actual experiments (data type: ``actual``)
* virtual experiments (data type: ``virtual``)

Actual experiments utilize the dual-detector setup for data acquisition and virtual experiments
synthetize dual-detector data from single-camera experiments.

With the actual dual-detector experimental data, XFusion first conducts spatial and temporal
calibration of the two image sequences.

* Temporal calibration is specific to the choice of the Phantom TMX7510 (Vision Research Inc., USA)
  camera and Shimadzu HPV-X2 (Shimadzu Corp., Japan) camera.
* Spatial calibration is based on the use of an electroformed copper grid
  (G400, Gilder Grids Ltd., UK).

At inference time, XFusion traverses the dual-camera image sequences in two ways: continuous and
double-interval.

* Under the continuous mode, high-quality xray images corresponding to each Shimadzu camera image
  are reconstructed.
* Under the double-interval mode, independent high-quality xray images corresponding to each Phantom
  camera image are reconstructed to evaluate performance metrics.

Initialization
==============

Run::

    xfusion init --model_type [EDVRModel or SwinIRModel] --data_type [virtual or actual]

After initialization, a configuration file named ``xfusion.conf`` will be generated in your home directory.
This configuration file is updated automatically during XFusion workflows.

Data preparation
================

Data for model pre-training
---------------------------

Download the Sharp dataset called `train_sharp <https://drive.google.com/open?id=1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz->`_
and Low Resolution dataset called
`train_sharp_bicubic <https://drive.google.com/open?id=1a4PrjqT-hShvY9IyJm3sPF0ZaXyrCozR>`_
from the `REDS dataset <https://seungjunnah.github.io/Datasets/reds>`_ to the directories specified in the
``convert`` section of the configuration file.

Data for model fine tuning
--------------------------

Fine tuning data are not available at this moment.

Data for testing
----------------

There are two sets of sample data to be downloaded from the
`Tomobank <https://tomobank.readthedocs.io/en/latest/source/data/docs.data.radio.html>`_.

Data conversion
===============

To convert the REDS data to gray-scale, run::

    xfusion convert \
      --dir-lo-convert [directory/to/low resolution/RGB/training image] \
      --dir-hi-convert [directory/to/high resolution/RGB/training image] \
      --out-dir-lo [directory/to/low resolution/gray-scale/training image] \
      --out-dir-hi [directory/to/high resolution/gray-scale/training image]

Training
========

To train the model with data from the virtual experiments, run::

    xfusion train \
      --dir-lo-train [directory/to/low resolution/gray-scale/training image] \
      --dir-hi-train [directory/to/high resolution/gray-scale/training image] \
      --dir-lo-val [directory/to/low resolution/gray-scale/validation image] \
      --dir-hi-val [directory/to/high resolution/gray-scale/validation image] \
      --opt [directory/to/training setting/yaml file] \
      --path-train-meta-info-file [directory/to/training image/meta data] \
      --path-val-meta-info-file [directory/to/validation image/meta data] \
      --pretrain_network_g [path/to/model weight/file/for/model initialization]

To train the model with data from the actual experiments, run::

    xfusion train \
      --dir-lo-train [directory/to/low resolution/gray-scale/paired training image] \
      --dir-hi-train [directory/to/high resolution/gray-scale/paired training image] \
      --dir-lo-val [directory/to/low resolution/gray-scale/paired validation image] \
      --dir-hi-val [directory/to/high resolution/gray-scale/paired validation image] \
      --dataroot_context [directory/to/low resolution/gray-scale/training image full] \
      --opt [directory/to/training setting/yaml file] \
      --path-train-meta-info-file [directory/to/training image/meta data] \
      --path-val-meta-info-file [directory/to/validation image/meta data] \
      --meta_info_context [directory/to/paired/high/lo/resolution/training image/index map] \
      --pretrain_network_g [path/to/model weight/file/for/model initialization]

Train calibration model
=======================

To train the calibration model with data from the virtual experiments (pretrain), run::

    xfusion train_calibration \
      --image_root [root/directory/to/training images] \
      --meta_info_file [directory/to/training image/meta data] \
      --evaluator_weights [path/to/spatiotemporal/fusion/model weight/file/for/training loss] \
      --flow_weights None \
      --full_weights None

To train the calibration model with data from the actual experiments (fine tune), run::

    xfusion train_calibration \
      --image_root [root/directory/to/training images] \
      --meta_info_file [directory/to/training image/meta data] \
      --evaluator_weights [path/to/spatiotemporal/fusion/model weight/file/for/training loss] \
      --full_weights [path/to/pretrained/calibration/model weight/file/for/model initialization]

Test data download
==================

To download test data, run::

    xfusion download \
      --dir-inf [tomobank/link/address/of/test/dataset] \
      --out-dir-inf [directory/to/testing image]

Spatial calibration
===================

Run::

    xfusion calibrate \
      --cal_model_file [path/to/calibration/model file] \
      --cal_dir [root/directory/to/calibration/experiment/data] \
      --cal_id [calibration experiment subdirectory name]

Inference
=========

For virtual experiment data, run::

    xfusion inference \
      --opt [directory/to/testing dataset/setting/yaml file] \
      --arch-opt [directory/to/model architecture/yaml file] \
      --img-class [dataset1 or dataset2] \
      --model_file [path/to/model file] \
      --machine [tomo or polaris]

For actual experiment data, run::

    xfusion inference \
      --arch-opt [directory/to/model architecture/yaml file] \
      --img-class [name of high-speed experiment] \
      --model_file [path/to/model file] \
      --traverse_mode [double or continuous] \
      --tfm_file [path/to/rigid/body/spatial/transformation/file] \
      --meta_file_scale [path/to/nonuniform/scaling/parameters] \
      --input_dir [root/directory/to/high/speed/experiment/data] \
      --case_list [experiment subdirectory names]

Suggested usage
===============

* For virtual experiment data, currently XFusion works for EDVRModel in the single-process mode and
  SwinIRModel in the multi-process mode. When working on the virtual experiment data, first initialize
  the software, and then do the inference.

* For actual experiment data, XFusion works for both models in a unified way. Do an independent
  initialization every time the model (EDVR-STF or SWIN-XVR) or the type of data (virtual or actual)
  changes. When working on the actual experiment data, first initialize the software, then do the
  spatial calibration, and last do the inference. Multiple experiments sharing the same spatial
  calibration can be grouped to run the model inference.
