# Loan Approval Prediction Project

## Introduction

This project was developed as part of the **Introduction to Programming** course. It applies Machine Learning (ML) techniques to financial decision-making processes, specifically focusing on loan approval predictions. The dataset consists of qualitative data on past loan applications, and the main objectives of the project are:

1. To analyze and process customer data.
2. To train and evaluate three ML models to predict whether a customer will be eligible for a loan.

The task is a binary classification problem, where the goal is to classify input data into one of two categories:
- **Y (Loan Approval)**
- **N (Loan Rejection)**

## Machine Learning Models Used

### 1. K-Nearest Neighbor (K-NN)
- **Accuracy**: 85.90%
- **Strengths**:
  - Performs well in predicting loan approval outcomes.
  - Handles positive loan approvals correctly.
- **Weaknesses**:
  - Misclassifies negative cases (loan rejections).
  - May require further tuning of distance metrics or scaling techniques.
- **Recommendations**:
  - Conduct cross-validation for stability.
  - Experiment with fine-tuning hyperparameters to improve reliability.

### 2. Logistic Regression (LR)
- **Accuracy**: 80.77%
- **Strengths**:
  - Provides a baseline performance.
  - Easy to interpret and implement.
- **Weaknesses**:
  - Low Area Under the ROC Curve (AUC) of 0.52, indicating poor rejection classification.
- **Recommendations**:
  - Introduce feature engineering (e.g., interaction terms or polynomial features).
  - Use regularization techniques or cost-sensitive learning for better optimization.

### 3. Decision Tree (DT)
- **Accuracy**: 75.64%
- **Strengths**:
  - Provides interpretability of results through feature importance analysis.
- **Weaknesses**:
  - Tends to overfit and struggles with false positives and negatives.
- **Recommendations**:
  - Optimize the model with parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.
  - Balance the tree structure to reduce overfitting.

## Dataset

The dataset contains qualitative data on past loan applications, including customer information and loan statuses. The preprocessing and analysis steps include:
- Data cleaning and encoding of categorical variables.
- Splitting the dataset into training and testing sets.
- Scaling or normalizing features as required for ML models.

## Dependencies

The project utilizes the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `os`
- `sklearn`

## File Descriptions

- **DABN22_Project.ipynb**: The Jupyter notebook containing the project code, analysis, and results.
- **loan_data_set.csv**: The dataset used for training and evaluating the models.
- **my_functions.py**: Custom Python functions used in the project for preprocessing and analysis.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
2. Navigate to the project directory:
   ```bash
   cd your_repository

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
5. Open the Jupyter notebook and run the cells:
  ```bash
   jupyter notebook DABN22_Project.ipynb

Results
The results of the ML models are summarized as follows:

K-NN achieved the highest accuracy (85.90%) but struggled with rejecting loans.
Logistic Regression provided a baseline but requires feature engineering for improvement.
Decision Tree showed moderate accuracy and high interpretability but tended to overfit.
Conclusion
The project demonstrates the application of ML techniques to financial processes, highlighting the strengths and weaknesses of the algorithms in a real-world context. Further improvements, such as better feature engineering and hyperparameter tuning, can significantly enhance the models' performance.
