README.md
### README for Breast Cancer Prediction Project

# Breast Cancer Prediction Project

## Introduction

Breast cancer is one of the most prevalent cancers affecting women worldwide. Early detection and accurate diagnosis are crucial for effective treatment and improved survival rates. This project aims to develop and evaluate machine learning models to predict breast cancer using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Dataset Description

The WDBC dataset contains 569 instances with 32 attributes:
- ID number
- Diagnosis (M = malignant, B = benign)
- 30 real-valued features computed for each cell nucleus (mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)

The dataset is publicly available at the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29).

## Dataset
The dataset used for the analysis is contained in this repository, named wdbc.csv

## Project Structure

1. **Introduction**
   - Overview of the project and its objectives.

2. **Data Description**
   - Detailed description of the dataset and its features.

3. **Data Cleaning**
   - Steps taken to clean the data, including handling missing values and dropping unnecessary columns.

4. **Exploratory Data Analysis (EDA)**
   - Visualizations and analyses to understand the distribution and relationships of features.

5. **Feature Engineering**
   - Encoding and scaling of features to prepare them for modeling.

6. **Feature Selection**
   - Selection of relevant features based on their correlation with the target variable.

7. **Model Development**
   - Development and evaluation of multiple machine learning models:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - Support Vector Classifier (SVC)
     - K-Nearest Neighbours (KNN) Classifier

8. **Results and Discussion**
   - Comparison of model performance using accuracy, precision, recall, F1 score, and ROC AUC metrics.
   - Detailed discussion on the performance of each model.

9. **Confusion Matrix**
   - Confusion matrices for each model to visualize correct and incorrect predictions.

10. **Model Deployment**
    - Deployment of the best-performing model (Logistic Regression) using Streamlit for real-time predictions.

## Results Summary

Model | Accuracy | Precision | Recall | F1 Score | ROC AUC
--- | --- | --- | --- | --- | ---
Logistic Regression | 0.964912 | 0.953488 | 0.953488 | 0.953488 | 0.962660
Decision Tree | 0.938596 | 0.950000 | 0.883721 | 0.915663 | 0.927776
Random Forest | 0.956140 | 0.952381 | 0.930233 | 0.941176 | 0.951032
Gradient Boosting | 0.956140 | 0.952381 | 0.930233 | 0.941176 | 0.951032
Support Vector Machine | 0.956140 | 0.975000 | 0.906977 | 0.939759 | 0.946446
K-Nearest Neighbours | 0.964912 | 0.975610 | 0.930233 | 0.952381 | 0.958074

## Model Deployment

The Logistic Regression model, which demonstrated the highest performance among the evaluated models, was successfully deployed using Streamlit. Streamlit is an open-source app framework that allows for the creation of interactive web applications. This deployment enables users to input relevant features and obtain real-time predictions on whether a breast cancer diagnosis is benign or malignant.

## How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/ola-stathub/aptech-project.git
   cd breast-cancer-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to use the application.

## Acknowledgments

This project utilizes the Wisconsin Diagnostic Breast Cancer (WDBC) dataset available at the UCI Machine Learning Repository. Special thanks to the authors of the dataset for making it publicly available.

## Contact

For any questions or feedback, please reach out to:
- Name: Oladele Ajayi
- Email: ajayioladeleb@gmail.com

THANK YOU

