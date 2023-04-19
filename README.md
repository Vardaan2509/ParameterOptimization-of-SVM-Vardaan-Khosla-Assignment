# ParameterOptimization-of-SVM-Vardaan-Khosla-Assignment
This is an assignment based on Parameter Optimization of SVM(Support Vector Machine) submitted for the course UCS654(Predictive Analytics using Statistics) by Vardaan Khosla.

For the code please refer to - https://github.com/Vardaan2509/ParameterOptimization-of-SVM-Vardaan-Khosla-Assignment/blob/main/Paramater%20Optimization_Vardaan%20Khosla.ipynb

# Parameter Optimization
Parameter optimization is the process of finding the best combination of hyperparameters for a machine learning model. Hyperparameters are parameters that are set before the learning process begins and cannot be learned from the data.The choice of hyperparameters can have a significant impact on the performance of the model, and parameter optimization is the process of finding the combination of hyperparameters that gives the best results. 

# Dataset Used - https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

The dataset used in this code is the "Dry Bean Dataset" which is available in the UCI Machine Learning Repository. The dataset contains information about different varieties of dry beans that are commonly consumed in Turkey, including 13611 samples of 7 different types of dry beans.
Each sample has 16 features that describe the physical characteristics of the beans, such as area, perimeter, compactness, length, width, etc. The target variable for the dataset is the type of bean, which is classified into one of the 7 types.

The purpose of this assignment is to build a Support Vector Machine (SVM) classifier to accurately predict the type of dry bean based on the physical characteristics of the bean. The code uses systematic sampling to generate 10 different samples from the testing set, and then trains multiple SVM models on each sample to identify the best hyperparameters.The step wise explanation of what actually we need to do is as follows:

Step 1 - Preprocess the data: Scale the features in the dataset using StandardScaler() function and store the scaled features in the "X" variable. Extract the target variable "Class" from the dataset and store it in the "y" variable.

Step 2 - Split the dataset into training and testing sets using train_test_split() function. Store the training and testing sets of features and target variables in X_train, X_test, y_train, and y_test respectively. Use a test size of 0.3 and a random state of 42.

Step 3 - generate 10 samples from the testing set using systematic sampling, where each sample is a tuple of (X_train, X_test_sample, y_train, y_test_sample). Append each sample to the "samples" list.

Step 4 - Loop through each sample in the "samples" list. For each sample, loop through the different kernel types ('linear', 'poly', 'rbf', 'sigmoid'). Generate random values for the hyperparameters C and gamma using np.random.uniform() function. Evaluate the accuracy of the SVM model with the current kernel and hyperparameters using the fitnessFunction() function. If the accuracy is better than the best accuracy so far, update the best accuracy and the corresponding best hyperparameters.

Step 5 - Store the results of hyperparameter tuning in a pandas DataFrame "result". Each row in the DataFrame contains the sample number, best accuracy, best kernel, best Nu, and best Epsilon.

Step 6 - Plot the convergence curve: Get the index of the sample with the highest accuracy from the "result" DataFrame and plot the convergence curve with the number of iterations on the x-axis and the accuracy on the y-axis, using matplotlib.pyplot functions.

![image](https://user-images.githubusercontent.com/114947593/233179250-c7db80f7-5122-464c-af3e-43014582dcf3.png)


![image](https://user-images.githubusercontent.com/114947593/233178969-9b0ea45f-2da1-4cf3-8129-c82b1f5c02c5.png)

# Result
The best accuracy is being achieved for sample number 6 which is equal to 0.95.

The corresponding hyperparameters are : Kernel = poly, Nu = 0.63, Epsilon = 0.18


