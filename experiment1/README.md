**********************************************from Sina


# How to run 
1) get a 64G pod on python 2:
launch-pytorch-gpu.sh -m 64

2) clone this folder on your account :
clone git address

3)
cd Experiment1

4)
sh tools/download.sh
sh tools/process.sh


5) run these 2 lines to get pytorch 0.3.0 :
conda create -n env python=2.7 mkl=2018 pytorch=0.3.0 -c pytorch -c intel
source activate env

6) now you are inside env istall these libraries:
pip install --user pillow
pip instal --user h5py

7) run the main :
pyhton main.py

Finish.



### Prerequisites

Make sure you are on a machine with a NVIDIA GPU and Python 2 with about 70 GB disk space.

1. Install [PyTorch v0.3](http://pytorch.org/) with CUDA and Python 2.7.
2. Install [h5py](http://docs.h5py.org/en/latest/build.html).


***********************************************************from main page:


## Bottom-Up and Top-Down Attention for Visual Question Answering

An efficient PyTorch implementation of the winning entry of the [2017 VQA Challenge](http://www.visualqa.org/challenge.html).

The implementation follows the VQA system described in "Bottom-Up and
Top-Down Attention for Image Captioning and Visual Question Answering"
(https://arxiv.org/abs/1707.07998) and "Tips and Tricks for Visual
Question Answering: Learnings from the 2017 Challenge"
(https://arxiv.org/abs/1708.02711).

## Results

| Model | Validation Accuracy | Training Time
| --- | --- | -- |
| Reported Model | 63.15 | 12 - 18 hours (Tesla K40) |
| Implemented Model | **63.58** | 40 - 50 minutes (Titan Xp) |

The accuracy was calculated using the [VQA evaluation metric](http://www.visualqa.org/evaluation.html).

## About

This is part of a project done at CMU for the course 11-777
Advanced Multimodal Machine Learning and a joint work between Hengyuan Hu,
Alex Xiao, and Henry Huang.

As part of our project, we implemented bottom up attention as a strong VQA baseline. We were planning to integrate object
detection with VQA and were very glad to see that Peter Anderson and
Damien Teney et al. had already done that beautifully.
We hope this clean and
efficient implementation can serve as a useful baseline for future VQA
explorations.

## Implementation Details

Our implementation follows the overall structure of the papers but with
the following simplifications:

1. We don't use extra data from [Visual Genome](http://visualgenome.org/).
2. We use only a fixed number of objects per image (K=36).
3. We use a simple, single stream classifier without pre-training.
4. We use the simple ReLU activation instead of gated tanh.

The first two points greatly reduce the training time. Our
implementation takes around 200 seconds per epoch on a single Titan Xp while
the one described in the paper takes 1 hour per epoch.

The third point is simply because we feel the two stream classifier
and pre-training in the original paper is over-complicated and not
necessary.

For the non-linear activation unit, we tried gated tanh but couldn't
make it work. We also tried gated linear unit (GLU) and it works better than
ReLU. Eventually we choose ReLU due to its simplicity and since the gain
from using GLU is too small to justify the fact that GLU doubles the
number of parameters.

With these simplifications we would expect the performance to drop. For
reference, the best result on validation set reported in the paper is
63.15. The reported result without extra data from visual genome is
62.48, the result using only 36 objects per image is 62.82, the result
using two steam classifier but not pre-trained is 62.28 and the result
using ReLU is 61.63. These numbers are cited from the Table 1 of the
paper: "Tips and Tricks for Visual Question Answering: Learnings from
the 2017 Challenge". With all the above simplification aggregated, our
first implementation got around 59-60 on validation set.

To shrink the gap, we added some simple but powerful
modifications. Including:

1. Add dropout to alleviate overfitting
2. Double the number of neurons
3. Add weight normalization (BN seems not work well here)
4. Switch to Adamax optimizer
5. Gradient clipping

These small modifications bring the number back to ~62.80.  We further
change the concatenation based attention module in the original paper
to a projection based module. This new attention module is inspired by
the paper "Modeling Relationships in Referential Expressions with
Compositional Modular Networks"
(https://arxiv.org/pdf/1611.09978.pdf), but with some modifications
(implemented in attention.NewAttention).  With
the help of this new attention, we boost the performance to ~63.58,
surpassing the reported best result with no extra data and less
computation cost.

