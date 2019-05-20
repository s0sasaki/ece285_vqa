# ECE285_VQA

Gupta Github Page: https://github.com/Shivanshu-Gupta/Visual-Question-Answering
Cadene Github Page: https://github.com/Cadene/vqa.pytorch

All papers with implemetation: https://paperswithcode.com/task/visual-question-answering

Explanation about VQA Dataset: https://visualqa.org/download.html   
VQA API to handle the dataset https://github.com/GT-vision-lab/VQA




./vqaTools

This directory contains the Python API to read and visualize the VQA dataset
vqaDemo.py (demo script)
vqaTools (API to read and visualize data)

./vqaEvaluation

This directory contains the Python evaluation code
vqaEvalDemo.py (evaluation demo script)
vqaEvaluation (evaluation code)   
   
To make torch work:   
export PATH=${PATH}:/datasets/torch/install/bin
   
export PATH=${PATH}:$(find /datasets/torch/install -type d | tr '\n' ':' | sed 's/:$//')
