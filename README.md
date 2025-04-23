# medmnist_classification

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&color=gray)  

A simple template for building and training deep learning models using PyTorch. This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author :** Kaptios <br />

## Overview

This repository contains a framework built on top of [PyTorch](https://pytorch.org/) and inspired by the deep learning framework from the Institute of Machine Learning in Biomedical Imaging : https://github.com/compai-lab/iml-dl. It abstracts away boilerplate code for training, evaluation, and inference tasks while still offering the flexibility of PyTorch for custom modifications. 

It supports the following:
- Model training pipelines (with data preprocessing)
- Experiment tracking (integration with Weights & Biases)
- Model evaluation 

## Installation

First, the dependencies should be installed the provided conda environment depending on your OS : 

```bash
conda env create -f environment.yml
conda activate torch-env
```

## Experiment tracking

By default this template uses WandB as a logging system and tracker for experiment metrics.
To enable this support you should have a free account at [wandb.ai](https://wandb.ai) and login with the CLI using :

```bash
wandb login
```

The CLI will ask for the API key that can be found in your wandb account page.


## Running a training task

You can run an task by pointing to its configuration file like :

```bash
python main.py --config_path ./tasks/default/train.yaml
```


## Export a saved model

The framework supports exporting a saved PyTorch model to ONNX.
To do so, an export config yaml file should be given as flag to the ```export.py``` script.

```bash
python export.py --export_config_path ./tasks/default/export.yaml
```

This file should look like : 

```yaml
export_path: './exports'
model_path: './saved_models/best_model.pth'
quantization_dataset:
  module_name: src.data.sets.super_resolution
  class_name: FastMRISuperResolutionDataReader
  parameters:
    data_folder: /path/to/dataset
    num_samples: 100
model:
  class_name: SRResUNet
  module_name: src.models.super_resolution
  parameters:
```

By default, the export also saves a quantized version of the model. For this to work, a Calibration Dataset should be passed using the ```quantization_dataset``` key.

### Create a custom task

You can define your own tasks by simply following the structure of the default task folder.
Alternatively, if the synforge CLI is installed you can use it to create the necessary files for you :

```bash
synforge generate task
```
