# Description 

This is project ECE285_VQA developed by team SAMS composed of Arda Bati, Marjan Emadi, So Sasaki, and Sina Shahsavari. The repository mainly consists of two implementations of Visual Question Answering (VQA). The experiment1 is an implementation for Bottom-Up and Top-Down Attention for VQA, which our final report is based on. The experiment2 is a completely different implementation for a vanilla VQA. The details are on the README file in each experiment directory.  

# Requirements and Usage

### Experiment1

The experiment1 requires 64G memory. To get sufficient resource on ing6 server, create a pod as follows:
```
launch-pytorch-gpu.sh -m 64
```

Use python 2.7 and install packages pillow and h5py.
```
conda create -n envname python=2.7 mkl=2018 pytorch=0.3.0 -c pytorch -c intel
pip install --user pillow h5py
```

To train the model, run the followings:
```
cd experiment1
sh tools/download.sh
sh tools/process.sh
python main.py
```

For demonstration, run the demo script on jupyter notebook:
```
experiment1/demo/Demo.ipyenb
```


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


# Code organization 

 - experiment1: An implementation for Bottom-Up and Top-Down Attention for VQA
 - experiment1/main.py: 
 - . . . 
 - experiment2: An implementation for a vanilla VQA.
 - experiment2/main.py: Main script to manage the training and validation of the model
 - experiment2/train.py: Module for training and validation
 - experiment2/vqa.py:  Module for the neural network architecture
 - experiment2/dataset.py: Module for data loading
 - experiment2/config.yml: Configuration file
 - experiment2/requirements.txt: List of the requirements
 - experiment2/README.md: Other information
 - misc: Miscellaneous scripts
 - misc/data_format_check.ipynb: Script for preliminary data visualization 
 - misc/some_useful_codes: Scripts which were not used after all

