# TurbFPNet: Neural Far‐Field Fourier Ptychography Imaging and A Real-world Benchmark

Our TurbFPNet is a two-term mixture framework, FP-P2S (see model_FP-P2S.py) for image generation and PRT (see model_PRT.py) for image restoration. 


## Simulation Data Preparation

Please follow this (https://data.vision.ee.ethz.ch/cvl/DIV2K/) to download the DIV2K training dataset.

Please follow this (https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to download test datasets (Set5/Set14/BSD100/Urban100).

# Running

**simulation for training data**
```python model_FP-P2S.py --ori_path {path_to_input_folder_for_simulation}```

we provided default parameters in the document (model_FP-P2S.py) for image generation and other applications can further optimize performance by modifying the simulation parameters for FP-P2S relevant to the coherent imaging system.

**Inference on simulated data**
```python test_image_TurbFPNet_simulation.py --model_name {path_to_model_for_image_restoration} --image_path {path_to_input_folder_for_sub-aperture_images} --image_name {image_prefix} --save_path {path_to_save_the_results}```

we provided default parameters in the document (test_image_TurbFPNet_simulation.py) for testing the restoration results of TurbFPNet on the simulated data.

**Inference on experimantal data**
```python test_image_TurbFPNet_exp.py --model_name {path_to_model_for_image_restoration} --image_path {path_to_input_folder_for_sub-aperture_images} --image_name {image_prefix} --save_path {path_to_save_the_results}```

we provided default parameters in the document (test_image_TurbFPNet_exp.py) for testing the restoration results of TurbFPNet on the experimental data. 

## Acknowledgement

https://github.com/Riponcs/TurbulenceSimulatorPython

https://github.com/XPixelGroup/BasicSR

https://github.com/whai362/PVT
