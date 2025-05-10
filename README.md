
This project explores the application of machine learning techniques to predict whether individuals may have difficulty living independently, using a subset of the 2023 American Community Survey (ACS) focused on Massachusetts residents.

## Dataset Overview
- Source: 2023 American Community Survey (ACS)
- Size: 4,318 rows and 117 columns (reduced to 70 after preprocessing)
- Target: `Class` (1 = difficulty living independently, 0 = no difficulty)

## Preprocessing Summary
- Removed low-variance and administrative columns
- Categorized features into nominal, ordinal, interval, and ratio types
- Context-based imputation of missing values
- Combined military service indicators into a single binary column
- Addressed multicollinearity by removing highly correlated features (correlation > 0.7)

## Class Imbalance Handling
Two resampling techniques were used to address class imbalance:
- SMOTE (Synthetic Minority Oversampling Technique)
- ROSE (Random Over-Sampling Examples)

## Feature Selection Methods
- Information Gain
- Chi-Square Test
- Principal Component Analysis (PCA)

## Classification Models
Each of the six datasets (resulting from three feature selection methods applied to two balanced datasets) was used to train the following models:
- Logistic Regression
- Support Vector Machines (SVM)
- Bagging
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM

## Evaluation Metrics
- F1-score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Cohen’s Kappa
