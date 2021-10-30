# MachineLearningPhw1-2
This program is machine Learning assignment programming homework 1, 2

(1) PHW-1
phw-1 file is find the best model and best parameters and best scalers and best encoder

The dataset is breast-cancer dataset 
Id and feature1~9 are feature attributes, target is target value (categorical column)
Some nan value exists in feature 6 (ex)…?)->So, feature6 is not integer object type.
16 values are ‘?’ So, we replace ‘?’ to np.nan value. (Because its’s trash data)

<img width="230" alt="캡처2" src="https://user-images.githubusercontent.com/74089524/139522849-8df46b3b-9827-4a8f-8a48-e426dadb9caa.PNG">

After preprocessing

<img width="252" alt="캡처1" src="https://user-images.githubusercontent.com/74089524/139522848-c5d9c97e-7e8a-4685-8bc1-ade97d447e04.PNG">

 I use model: decision Tree with gini index, entrophy, support vector machine, logistic regression and normalizer, standard scaler, maxabs scaler, robust scaler, min max scaler, label encoder, ordinal encoder
 
 First, i made the function with each model
 
 like this:
 def decisionTreeWithGini(x,y):
 x is the features and y is the target
 
 The whol structure is 
 def bestOfbest(x,y):
 
 1. we put the feature and target
 2. they divide train and test dataset
 3.  they execute at each model and they measure the score of best model 
 4.  they tried the all scaler and encoder combination and return the best combination

Output:

<img width="465" alt="캡처3" src="https://user-images.githubusercontent.com/74089524/139522958-44762c9c-282e-41d7-b917-73a4682b8550.PNG">
