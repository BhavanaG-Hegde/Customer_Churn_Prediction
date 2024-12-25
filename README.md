Customer Churn Prediction

Description:
This project predicts customer churn for a telecom company using machine learning models, specifically Random Forest, Logistic Regression, and Decision Tree. The goal is to predict whether a customer will leave the company based on various factors like age, balance, and membership status.

Table of Contents:
1. Introduction
2. Installation Instructions
3. Usage
4. Acknowledgements

Introduction:
This project uses a dataset that includes customer data from a telecom company, such as account balance, age, and activity status, to predict whether a customer will leave the company (churn). The project implements different machine learning models, including Random Forest and Logistic Regression to make accurate predictions.

Key features of the project:
- Data preprocessing and feature engineering
- Model training and evaluation using metrics like accuracy and ROC-AUC
- Ability to save and load trained models for future use

Installation Instructions:
1. Clone the repository:
    git clone https://github.com/BhavanaG-Hegde/Customer_Churn_Prediction.git
2. Navigate to the project directory:
    cd Customer_Churn_Prediction
3. Install the required dependencies:
    pip install -r requirements.txt
4. Ensure that you have the required dataset, Modified_Churn_Modelling.csv, in the project folder.

Usage:
To run the churn prediction models:

1. Train the Random Forest model:
   - Execute the following Python script:
     python train_random_forest.py
   - The Random Forest model will be trained, and the results (accuracy, ROC-AUC score) will be displayed.
   - The trained Random Forest model and scaler will be saved to files for future use:
     - random_forest_model.pkl
     - scaler_1.pkl

2. Train the Logistic Regression model:
   - Execute the following Python script:
     python train_logistic_regression.py
   - The Logistic Regression model will be trained, and the results (accuracy, ROC-AUC score) will be displayed.
   - The trained Logistic Regression model and scaler will be saved to files for future use:
     - log_reg_model.pkl
     - scaler.pkl

3. You can modify the dataset or change the features as needed by editing the corresponding Python script (train_random_forest.py or train_logistic_regression.py).

Acknowledgements:
- Scikit-learn: For implementing the machine learning models.
- Pandas: For data manipulation and preprocessing.
- Matplotlib/Seaborn: For visualizing results and model performance.
- GitHub: For hosting the project.

