=====
About
=====

`XFusion <https://github.com/xray-imaging/XFusion>`_ implements deep learning-based spatiotemporal fusion (STF) for high-fidelity ultra-high-speed (UHS) full-field x-ray radiography.

Full-field UHS x-ray imaging experiments are widely used to characterize fast processes and dynamic phenomena. In many experiments, x-ray videos can be acquired using distinct detector/camera configurations that offer complementary strengths (e.g., one sequence with higher spatial resolution but lower frame rate, and another with higher frame rate but lower spatial resolution). However, the scientific potential of jointly using these complementary acquisitions is often not fully exploited.

XFusion addresses this by using a deep learning-based STF framework to fuse two complementary x-ray image sequences and reconstruct a target sequence that simultaneously achieves:

* high spatial resolution,
* high frame rate, and
* high fidelity.

In the associated work, a transfer learning strategy is used to train the model, and performance is evaluated using standard image-quality metrics including peak signal-to-noise ratio (PSNR), average absolute difference (AAD), and structural similarity (SSIM). The STF approach is compared against multiple alternatives, including a baseline deep learning model, a Bayesian fusion framework, and bicubic interpolation, and is shown to outperform these methods across different input frame separations and noise levels.

As an example configuration, using 3 subsequent images from a low-resolution (LR) sequence (4× lower spatial resolution) together with 2 images from a high-resolution (HR) sequence (20× lower frame rate), the proposed approach achieved average PSNR values of 37.57 dB and 35.15 dB (on two independent datasets).

When coupled with an appropriate combination of high-speed camera systems, STF-based reconstruction can enhance the performance—and therefore the scientific value—of UHS x-ray imaging experiments.

Implementation notes
====================

This implementation builds on the `BasicSR toolbox <https://github.com/XPixelGroup/BasicSR>`_.
For model pre-training, data are collected from the `REDS dataset <https://seungjunnah.github.io/Datasets/reds>`_.
