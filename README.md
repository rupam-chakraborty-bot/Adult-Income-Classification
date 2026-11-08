# Adult-Income-Classification
Project Name : Adult Income Classification using Machine Learning
Author: Rupam Chakraborty
BITS ID: 2026ak05501

**A. Problem Statement**

The objective of this project is to predict whether an individual's annual income is greater than $50,000 based on demographic, educational, and employment-related information.

This is a Binary Classification problem where:

Income <=50K → Class 0
Income >50K → Class 1

Five Machine Learning classification algorithms were implemented, evaluated, and compared to identify the best-performing model for the Adult Income dataset.

**B. Dataset Description**
Dataset Name: Adult Income Dataset
Source: UCI Machine Learning Repository
Dataset Link: https://archive.ics.uci.edu

**Dataset Characteristics**
Total Records: 48,842
Number of Features: 14
Target Variable: income
Problem Type: Binary Classification
Target Classes
<=50K
50K

**Features Used**
    **Numerical Features**
      age
      fnlwgt
      education-num
      capital-gain
      capital-loss
      hours-per-week
    **Categorical Features**
      workclass
      education
      marital-status
      occupation
      relationship
      race
      sex
      native-country
**Data Preprocessing Steps**
The following preprocessing steps were performed:

  1. Dataset loaded directly from the UCI repository using Python.
  2. Missing values represented by '?' were replaced with NaN.
  3. Rows containing missing values were removed.
  4. Target labels were converted into numerical form.
  5. Categorical features were transformed using One-Hot Encoding.
  6. Data was split into training and testing sets.
  7. Feature scaling was performed using StandardScaler.

**D. Models Used and Performance Comparison**
The following machine learning models were implemented:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes Classifier
Support Vector Machine (SVM)

**Model Performance Comparison**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8444 | 0.9016 | 0.7326 | 0.5865 | 0.6515 | 0.5586 |
| **Decision Tree** | 0.8104 | 0.7471 | 0.6169 | 0.6204 | 0.6186 | 0.4925 |
| **KNN** | 0.8219 | 0.8471 | 0.6624 | 0.5740 | 0.6151 | 0.5021 |
| **Naive Bayes** | 0.5656 | 0.8067 | 0.3579 | 0.9474 | 0.5195 | 0.3520 |
| **Support Vector Machine** | 0.8456 | 0.9022 | 0.7399 | 0.5812 | 0.6510 | 0.5603 |

**E. Observations on Model Performance**
| ML Model Name | Observation About Model Performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression achieved strong overall performance with an Accuracy of 84.44% and an AUC of 0.9016. The model provided a good balance between precision and recall and served as a strong baseline classifier for the Adult Income dataset. |
| **Decision Tree** | Decision Tree achieved an Accuracy of 81.04%. The model was easy to interpret but showed lower overall performance compared to Logistic Regression and SVM. Its MCC and AUC scores indicate weaker generalization capability on this dataset. |
| **KNN** | KNN achieved an Accuracy of 82.19% and an AUC of 0.8471. The model performed reasonably well but was influenced by the choice of neighbors and feature scaling. Performance was better than Decision Tree in some metrics but lower than Logistic Regression and SVM. |
| **Naive Bayes** | Naive Bayes achieved the highest Recall (94.74%) among all models, meaning it successfully identified most positive income cases. However, its Precision (35.79%) and overall Accuracy (56.56%) were comparatively low, resulting in many false positive predictions. |
| **Support Vector Machine** | Support Vector Machine (SVM) delivered the best overall results with the highest Accuracy (84.56%), AUC (0.9022), Precision (73.99%), and MCC (0.5603). The model demonstrated strong predictive capability and the best overall balance of classification metrics. |
| **Overall Winner for Dataset** | Support Vector Machine (SVM) was selected as the best-performing model because it achieved the highest Accuracy, AUC, Precision, and MCC among all five classifiers, indicating superior overall classification performance on the Adult Income dataset. |
