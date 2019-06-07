# SAMS_VQA

## Description 

This is project SAMS_VQA developed by team SAMS composed of Arda Bati, Marjan Emadi, So Sasaki, and Sina Shahsavari. The repository mainly consists of two implementations of Visual Question Answering (VQA). The experiment1 is an implementation for Bottom-Up and Top-Down Attention for VQA, which our final report is based on. The experiment2 is a completely different implementation for a vanilla VQA. The details are on the README file in each experiment directory.  

## Requirements and Usage

### Experiment1

The experiment1 requires 64G memory. To get sufficient resource on ing6 server, create a pod as follows:
```
launch-pytorch-gpu.sh -m 64
```

Use python 2.7 and install packages pillow and h5py. Since CUDA (Version 8) and pytorch (0.3.1) of DSMLP Python 2.7 pod is imcompatible, you need to downgrade pytorch to 0.3.0. 
```
conda create -n envname python=2.7 mkl=2018 pytorch=0.3.0 -c pytorch -c intel
source activate envname
pip install --user pillow h5py
```

To train the model, run the followings:
```
cd experiment1
sh tools/download.sh
sh tools/process.sh
python main.py
```

For demonstration, you need the experiment results, our_answers.dms. This file is uploaded in experiment1/demo, but you can also generate it as follows:
```
cd experiment1/demo
python demo.py
```

Then run the demo script on jupyter notebook:

- experiment1/demo/Demo.ipyenb

### Experiment2

For experiment2, use python 3.7 and install packages torchtext, tensorboardX, and utils. To execute the code, run the followings:

```
cd experiment2
pip install --user -r requirements.txt
mkdir results
mkdir preprocessed
mkdir preprocessed/img_vgg16feature_train
mkdir preprocessed/img_vgg16feature_val
python main.py -c config.yml
```

To skip preprocessing after the first execution, disable 'preprocess' in the config file.

The experiment2 does not include demo scripts or trained model parameters.


## Code organization 

### experiment1

 - experiment1: An implementation for Bottom-Up and Top-Down Attention for VQA
 - experiment1/main.py: 
 - experiment1/train.py: 
 - experiment1/language_model.py: 
 - experiment1/fc.py: 
 - experiment1/dataset.py: 
 - experiment1/base_model.py: 
 - experiment1/attention.py: 
 - experiment1/utils.py: 
 - experiment1/demo: 
 - experiment1/demo/Demo.ipynb: 
 - experiment1/demo/demo.py: 
 - experiment1/demo/base_model.py: 
 - experiment1/demo/glove6b_init_300d.npy: 
 - experiment1/demo/model.pth: 
 - experiment1/demo/our_answers.dms: 
 - experiment1/demo/readme.md: 
 - experiment1/demo/test.dms: 
 - experiment1/demo/trainval_label2ans.pkl: 
 - experiment1/tools: 
 - experiment1/tools/compute_softscore.py: 
 - experiment1/tools/create_dictionary.py: 
 - experiment1/tools/detection_features_converter.py: 
 - experiment1/tools/download.sh: 
 - experiment1/tools/process.sh: 
 - experiment1/data: 
 - experiment1/data/train_ids.pkl: 
 - experiment1/data/val_ids.pkl: 
 - experiment1/README.md: 

### experiment2

 - experiment2: An implementation for a vanilla VQA.
 - experiment2/main.py: Main script to manage the training and validation of the model
 - experiment2/train.py: Module for training and validation
 - experiment2/vqa.py:  Module for the neural network architecture
 - experiment2/dataset.py: Module for data loading
 - experiment2/config.yml: Configuration file
 - experiment2/requirements.txt: List of the requirements
 - experiment2/README.md: Other information

### Miscellaneous scripts

 - misc: Miscellaneous scripts
 - misc/data_format_check.ipynb: Script for preliminary data visualization 
 - misc/some_useful_codes: Scripts which were not used after all

