# Elevate-labs
# AI & ML Internship - Task 1: Titanic Data Cleaning & Preprocessing

## Project Overview
This project, developed as part of the AI & ML Internship, focuses on the essential steps of data cleaning and preprocessing for the classic Titanic dataset. The goal is to transform raw, messy data into a clean, structured, and normalized format suitable for machine learning model training.

## Dataset
The dataset used is the famous Titanic passenger data, which includes various features about passengers (e.g., age, gender, fare, class) and their survival status. The dataset was obtained from: [https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

## Key Steps Performed
The following data cleaning and preprocessing steps were executed:

1.  **Data Loading & Initial Exploration:**
    *   Loaded the dataset into a Pandas DataFrame.
    *   Displayed the first few rows to understand the structure.
    *   Checked data types, non-null counts (`df.info()`), and summary statistics (`df.describe()`).
    *   Identified columns with missing values (`df.isnull().sum()`).

2.  **Missing Value Handling:**
    *   **Age:** Imputed missing values with the *median* age to maintain distribution and reduce outlier sensitivity.
    *   **Embarked:** Imputed missing values with the *mode* (most frequent port of embarkation).
    *   **Cabin:** Filled missing values with the string 'Unknown' due to a large number of missing entries, treating 'Unknown' as a distinct category.

3.  **Categorical Variable Encoding:**
    *   Converted 'Sex' and 'Embarked' columns into numerical dummy variables using one-hot encoding (`pd.get_dummies`).
    *   `drop_first=True` was used to prevent multicollinearity.

4.  **Numerical Feature Normalization:**
    *   Standardized 'Age' and 'Fare' columns using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. This is crucial for many machine learning algorithms.

5.  **Outlier Detection & Removal (Fare Column):**
    *   **Visualization Before Removal:** A histogram was generated to visualize the distribution of the 'Fare' column, highlighting potential outliers.
    *   **IQR Method:** Outliers in the 'Fare' column were identified and removed using the Interquartile Range (IQR) method (values outside of Q1 - 1.5*IQR and Q3 + 1.5*IQR).
    *   **Visualization After Removal:** A second histogram was generated to demonstrate the distribution of 'Fare' after outlier removal, showing a more concentrated and normalized spread.

6.  **Correlation Analysis:**
    *   A correlation heatmap was generated for all numerical features to visualize relationships between variables, helping to identify strong positive or negative correlations.

7.  **Output:**
    *   The cleaned and preprocessed dataset was saved to a new CSV file named `cleaned_titanic.csv`.

## Files in this Repository
*   `titanic_data_cleaning.ipynb`: The Jupyter/Colab notebook containing all the code and analysis steps.
*   `cleaned_titanic.csv`: The output CSV file containing the cleaned and preprocessed Titanic dataset.
*   `README.md`: This description file.

## How to Run the Code
1.  **Open the Notebook:** You can open `titanic_data_cleaning.ipynb` directly in Google Colab or Jupyter Notebook.
2.  **Run Cells:** Execute each code cell sequentially to reproduce the data cleaning and preprocessing steps.
3.  **Inspect Outputs:** Review the printed outputs, visualizations, and generated `cleaned_titanic.csv` file.

## Libraries Used
*   `pandas` (for data manipulation and analysis)
*   `numpy` (for numerical operations)
*   `matplotlib.pyplot` (for plotting and visualizations)
*   `seaborn` (for enhanced statistical data visualizations)
*   `sklearn.preprocessing.StandardScaler` (for feature scaling)

## Conclusion
This project demonstrates a robust workflow for preparing real-world data for machine learning tasks, covering essential techniques from handling missing values to outlier treatment and feature scaling.
