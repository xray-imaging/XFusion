# XFusion
Deep learning-based spatiotemporal fusion for high-fidelity ultra-high-speed x-ray radiography  
A model to reconstruct high quality x-ray images by combining the high spatial resolution of high-speed camera and high temporal resolution of ultra-high-speed camera image sequences.  
Models and methods for spatial and temporal calibration are also included.

## Prerequisites
This implementation is based on the [BasicSR toolbox](https://github.com/XPixelGroup/BasicSR) and the [RAFT model](https://github.com/princeton-vl/RAFT). Data for model pre-training are collected from the [REDS dataset](https://seungjunnah.github.io/Datasets/reds).  

## Usage
### Package description
Currently, xfusion supports 2 model familities for high-quality xray image sequence reconstruction- the EDVR and Swin vision transformer, respectively.  
In addition, xfusion is functional with x-ray image sequences from both the actual (data type: actual) and virtual (data type: virtual) experiments.  
* Actual experiments utilize the dual-detector setup for data acquisition and virtual experiments synthetize dual-detector data from single-camera experiments.  

With the actual dual-detector experimental data, xfusion first conducts the spatial and temporal calibration of the two image sequences.  
* Temporal calibration is specific to the choice of the Phantom TMX7510 (Vision Research Inc., USA) camera and Shimadzu HPV-X2 (Shimadzu Corp., Japan) camera.  
* Spatial calibration is based on the use of an electroformed copper grid (G400, Gilder Grids Ltd., UK).  

Then at inference time, xfusion traverses the dual-camera image sequences in two ways: continuous and double-interval.  
* Under the continuous mode, high-quality xray images corresponding to each Shimadzu camera image are reconstructed.  
* Under the double-interval mode, independent high-quality xray images corresponding to each Phantom camera image are reconstructed to evaluate performance metrics.

### Package installation
Navigate to the project root directory and then run
```
pip install .
```
to install the package to the selected virtual environment.

### Initialization
Run
<pre>
xfusion init --model_type <i>[EDVRModel or SwinIRModel]</i> --data_type <i>[virtual or actual]</i>
</pre>
After initialization, a configuration file "xfusion.conf" will be generated in the home directory. This configuration file will be updated automatically within the workflow of the xfusion package.

### Data preparation
#### Data for model pretraining
Download the Sharp dataset called [train_sharp](https://drive.google.com/open?id=1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-) and Low Resolution dataset called [train_sharp_bicubic](https://drive.google.com/open?id=1a4PrjqT-hShvY9IyJm3sPF0ZaXyrCozR) from [REDS dataset](https://seungjunnah.github.io/Datasets/reds) to the directories specified in the "convert" section of the configuration file.
#### Data for model fine tuning
Fine tuning data are not available at this moment.
#### Data for testing
There are two sets of sample data to be downloaded from the [Tomobank](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.radio.html).

### Data conversion
To convert the REDS data to gray-scale, run
<pre>
xfusion convert --dir-lo-convert <i>[directory/to/low resolution/RGB/training image]</i> --dir-hi-convert <i>[directory/to/high resolution/RGB/training image]</i> --out-dir-lo <i>[directory/to/low resolution/gray-scale/training image]</i> --out-dir-hi <i>[directory/to/high resolution/gray-scale/training image]</i>
</pre>

### Training
To train the model with data from the virtual experiments, run
<pre>
xfusion train --dir-lo-train <i>[directory/to/low resolution/gray-scale/training image]</i> --dir-hi-train <i>[directory/to/high resolution/gray-scale/training image]</i> --dir-lo-val <i>[directory/to/low resolution/gray-scale/validation image]</i> --dir-hi-val <i>[directory/ti/high resolution/gray-scale/validation image]</i> --opt <i>directory/to/training setting/yaml file</i> --path-train-meta-info-file <i>[directory/to/training image/meta data]</i> --path-val-meta-info-file <i>[directory/to/validation image/meta data]</i> --pretrain_network_g <i>[path/to/model weight/file/for/model initialization]</i>
</pre>
To train the model with data from the actual experiments, run
<pre>
xfusion train --dir-lo-train <i>[directory/to/low resolution/gray-scale/paired training image]</i> --dir-hi-train <i>[directory/to/high resolution/gray-scale/paired training image]</i> --dir-lo-val <i>[directory/to/low resolution/gray-scale/paired validation image]</i> --dir-hi-val <i>[directory/ti/high resolution/gray-scale/paired validation image]</i> --dataroot_context <i>[directory/to/low resolution/gray-scale/training image full]</i> --opt <i>directory/to/training setting/yaml file</i> --path-train-meta-info-file <i>[directory/to/training image/meta data]</i> --path-val-meta-info-file <i>[directory/to/validation image/meta data]</i> --meta_info_context <i>[directory/to/paired/high/lo/resolution/training image/index map]</i> --pretrain_network_g <i>[path/to/model weight/file/for/model initialization]</i>
</pre>

### Train calibration model
To train the calibration model with data from the virtual experiments (pretrain), run
<pre>
xfusion train_calibration --image_root <i>[root/directory/to/training images]</i> --meta_info_file <i>[directory/to/training image/meta data]</i>
--evaluator_weights <i>[path/to/spatiotemporal/fusion/model weight/file/for/training loss]</i>
--flow_weights None --full_weights None
</pre>
To train the calibration model with data from the actual experiments (fine tune), run
<pre>
xfusion train_calibration --image_root <i>[root/directory/to/training images]</i> --meta_info_file <i>[directory/to/training image/meta data]</i>
--evaluator_weights <i>[path/to/spatiotemporal/fusion/model weight/file/for/training loss]</i>
--full_weights <i>[path/to/pretrained/calibration/model weight/file/for/model initialization]</i>
</pre>

### Test data
To download test data, run
<pre>
xfusion download --dir-inf <i>[tomobank/link/address/of/test/dataset]</i> --out-dir-inf <i>[directory/to/testing image]</i>
</pre>

### Spatial calibration
Run
<pre>
xfusion calibrate --cal_model_file <i>[path/to/calibration/model file]</i> --cal_dir <i>[root/directory/to/calibration/experiment/data]</i>
--cal_id <i>[calibration experiment subdirectory name]</i>
</pre>

### Inference
For virtual experiment data, run
<pre>
xfusion inference --opt <i>directory/to/testing dataset/setting/yaml file</i> --arch-opt <i>directory/to/model architecture/yaml file</i> --img-class <i>dataset1 or dataset2</i> --model_file <i>[path/to/model file]</i> --machine <i>tomo or polaris</i>
</pre>
For actual experiment data, run
<pre>
xfusion inference --arch-opt <i>directory/to/model architecture/yaml file</i> --img-class <i>name of high-speed experiment</i> --model_file <i>[path/to/model file]</i> --traverse_mode <i>double or continuous</i> --tfm_file <i>[path/to/rigid/body/spatial/transformation/file]</i> --meta_file_scale <i>[path/to/nonuniform/scaling/parameters]</i> --input_dir <i>[root/directory/to/high/speed/experiment/data]</i> --case_list <i>experiment subdirectory names</i>
</pre>

### Suggested usage
* For virtual experiment data, currently xfusion works for EDVRModel in the single-process mode and SwinIRModel in the multi-process mode.  
When working on the virtual experiment data, first initialize the software, and then do the inference.  

* For actual experiment data, xfusion works for both models in a unified way.  
Do an independent initialization every time the model (EDVR-STF or SWIN-XVR) or the type of data (virtual or actual) changes.  
When working on the actual experiment data, first initialize the software, then do the spatial calibration, and last do the inference.  
Multiple experiments sharing the same spatial calibration can be grouped to run the model inference.
