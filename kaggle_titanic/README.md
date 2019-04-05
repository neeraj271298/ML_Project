# TITANIC KAGGLE COMPETITION

Kaggle Competition Name = Titanic : Machine Learning From Disaster
Submission Name = Neeraj Bhardwaj

Dataset = Zip File contains three CSV files and one Python file.
1.	Test.csv  =  it contain test data
2.	Train.csv = it contain training data
3.	Submission.csv = it contain the result of test data 

Code  =  the code file is divided in three section Data Preprocessing , First Model ,
               Second Model . 

1.	Data Preprocessing = In this section the given data is preprocess like removing the Nan value by finding the average of column and replace it at place of Nan , converting categorical features into numbers , scaling the data , applying Principal Component Analysis for features extraction  

2.	First Model =   In this section we create a KNearestNeighbors Classification Model and train it with training dataset and finally we predict for test dataset . confusion matrix is used to find accuracy . K Fold Cross Validation is used to find the mean Accuracy over the different sets of training dataset . Grid Search is used to find the optimal value of parameters .

3.	Second Model = in this section we create an Artificial Neural Network Model with one hidden layer . We uses the Rectified Linear Unit (relu) activation function for input layer and hidden layer and Sigmoid activation function for ouput layer . This model is compiled with Adam optimizer and Binary Cross Entropy loss function .



Result   
                 Accuracy of KNN Model  =  92.11 %
	    Mean Accuracy of KNN Model        =  81.26 %
	    Accuracy of ANN Model             =  91.38% 
