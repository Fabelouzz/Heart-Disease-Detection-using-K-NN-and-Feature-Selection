Heart Disease Detection using K-NN and Feature Selection
This repository contains a project focused on detecting the presence of heart disease using a K-Nearest Neighbors (K-NN) classifier. The project highlights the importance of feature selection in improving the performance of machine learning models, particularly in medical datasets where some features may be irrelevant or detrimental to model accuracy.

Key Features:
K-Nearest Neighbors (K-NN) Classifier: A supervised learning algorithm used to classify the dataset into two classes: presence or absence of heart disease. The classifier is trained and tested on a medical dataset split into training and testing sets.

Feature Selection: The project employs two advanced feature selection techniques to identify the most relevant features:

Random-Restart Hill Climbing: This method starts with a random solution and iteratively explores the solution space by flipping bits representing features. It keeps track of the best solution found across multiple restarts to avoid local optima.
Random-Restart Variable Neighbor Search: Similar to hill climbing but with a more sophisticated approach to exploring neighboring solutions. This method adjusts the neighborhood size dynamically, expanding the search when improvements are found.
Performance Evaluation: Both algorithms are run 10 times, and the average accuracy of the classifier is reported to ensure robustness and reliability of the results. Accuracy is calculated based on the confusion matrix, which tracks true positives, true negatives, false positives, and false negatives.

Code Structure: The repository is organized with clear and well-documented Python code, making it easy to follow the steps for data preprocessing, feature selection, model training, and evaluation.
This project is a practical demonstration of applying machine learning techniques to real-world medical data, offering insights into feature selection and model optimization for better predictive accuracy.
