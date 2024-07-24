# Data Science Experiment 2

This repository contains the code and dataset for Experiment 2 of the Data Science Fundamentals with Python course.

## Description

In this experiment, we:
- Loaded the "Adult" dataset from the UCI Machine Learning Repository.
- Performed data preprocessing by cleaning the dataset.
- Converted categorical variables to dummy variables.
- Normalized numerical features.

## Files

- `experiment_2.ipynb`: The Jupyter Notebook containing the experiment code.
- `cleaned_adult_data.csv`: The cleaned dataset generated from the experiment.

## Usage

To run the code, open the `experiment_2.ipynb` file in Google Colab or Jupyter Notebook.

## Steps to Reproduce

1. **Set up Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Create a new notebook.

2. **Import Necessary Libraries**:
   - Start by importing the required libraries.

    ```python
    import pandas as pd
    import numpy as np
    ```

3. **Load the Dataset**:
   - Use the URL of the dataset from the UCI ML repository. For this example, let's use the "Adult" dataset.

    ```python
    # Load the dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
               'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
               'hours_per_week', 'native_country', 'income']
    data = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)
    ```

4. **Display the Dataset**:
   - Show the first few rows of the dataset to verify it loaded correctly.

    ```python
    data.head()
    ```

5. **Check for Missing Values**:
   - Identify and count missing values in the dataset.

    ```python
    data.isnull().sum()
    ```

6. **Drop Rows with Missing Values**:
   - Remove rows with missing values.

    ```python
    data_cleaned = data.dropna()
    data_cleaned.isnull().sum()
    ```

7. **Convert Categorical Variables**:
   - Use `pd.get_dummies` to convert categorical variables into dummy variables.

    ```python
    data_cleaned = pd.get_dummies(data_cleaned)
    data_cleaned.head()
    ```

8. **Normalize Numerical Features**:
   - Standardize numerical features using `StandardScaler`.

    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numerical_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])
    data_cleaned.head()
    ```

9. **Save and Export the Cleaned Dataset**:
    - Save the cleaned dataset to a CSV file and download it locally.

    ```python
    data_cleaned.to_csv('cleaned_adult_data.csv', index=False)
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
