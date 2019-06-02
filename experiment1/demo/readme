To run this Demo you have to opetions:

1. 
If you want to see the result of our trained network on some samples use this option:
There is a jupyter notebook that present result of our method on 6 random samples (4 succeeded and 2 failure). 
To Run this file you need to upload two files: 1) our_answers.dms file which contains the output of our model for these 6 test images(estimated scores to all possible answer from a generated dictionary in preprocessing) 2) train_label2ans.pkl which is a dictionary that will map scores to a answer for given question.
Note that: Images, question and correct answer will be extracted from VQA Dataset on DSMLP datasets folder, so their address must be same.

2. 
If you want to run the trained model on test images yourself you can use the below instruction:
First of all, you need to upload the whole demo folder. It contains the trained model (model.pth) which is about 100 MB, the extracted features of 6 randomly selected test images, the model architecture (to imitate a new model and then load state_dict of best model on it.) and the demo.py file that will send images through the best model and save the results on a file named our_answers.dms. Then you can use option one to see the results.

Note: Same as what mentioned in previous page for tuning the demo.py file since it needs to send the model to CUDA, you need to upgrade Cuda to version 9 or downgrade pythorch to 0.3.0 (for this you can use the instruction of prevues page which create an environment with lower version of pytorch)

(The problem is not about the architecture is about incompatibility of pytorch and cada version that cannot send a RNN model to CUDA)




