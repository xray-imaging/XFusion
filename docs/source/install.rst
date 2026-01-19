=======
Install
=======

This section covers the basics of how to install `XFusion <https://github.com/xray-imaging/XFusion>`_.

.. contents:: Contents
   :local:

Prerequisites
=============

* Python 3.9+ (examples below use conda)
* Git

Installing from source (conda)
==============================

Create and activate a dedicated conda environment::

    conda create -n xfusion python=3.9
    conda activate xfusion

Clone the repository::

    git clone https://github.com/xray-imaging/XFusion.git
    cd XFusion

Install XFusion::

    pip install .

Install runtime dependencies (as needed)
========================================

XFusion depends on a number of scientific Python packages. If you maintain a file
such as ``envs/requirements.txt`` and/or ``envs/requirements-doc.txt``, install those
dependencies as appropriate for your use case.

For example, with conda::

    conda install -c conda-forge numpy pyyaml

If you prefer pip for dependencies::

    pip install -r envs/requirements.txt

Test the installation
=====================

After installation, verify the command-line interface is available::

    xfusion --help

Update
======

To update your local checkout::

    cd XFusion
    git pull
    pip install .
