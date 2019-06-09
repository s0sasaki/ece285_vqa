## This is a Demo which represent the result of our trained network on some samples(4 succeeded and 2 failure).

Run Demo.ipynb to see the samples. This notebook extract 6 questions with their corresponded images and correct answers from VQA Dataset which is in DSMLP datasets folder and load output vector of the best trained model for these quesion from our_answers.dms file to show answers of network to the questions. 
(our_answers.dms contains estimated scores to all possible answer from a generated dictionary

You can also generate our_answers file by sending samples through forward path of trained model. first open a pod on python 2.7 and downgrade the pytorch version due to explaintion in the following note and run the demo.py file:

```
python demo.py
```




### Important Note:
There is an incompatibility between CUDA (Version 8) and pytorch (0.3.1) of DSMLP Python 2.7 pod that causes a problem when RNN models are being passed from CPU to CUDA (The problem is a general problem for current version of CUDA and pytorch on server in python 2.7 pods and it is not realted to the network architecture). To address this issue before running demo.py or training the model you need to downgrade the pytorch version to 0.3.0 To do this you can use the instructions from readme file of eaperiment 1 page which create an environment with lower version of pytorch.

```
conda create -n envname python=2.7 mkl=2018 pytorch=0.3.0 -c pytorch -c intel
source activate envname
```

An alternative to this might be upgrading CUDA from 8 to 9, but we did not try this method due to limiations on access to the server and therefore donot recommend it.






