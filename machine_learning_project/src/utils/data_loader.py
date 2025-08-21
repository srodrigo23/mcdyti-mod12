# data_loader.py

import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a specified CSV file path.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and selecting relevant features.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: A preprocessed DataFrame.
    """
    # Replace 'PrivacySuppressed' with NaN
    df.replace('PrivacySuppressed', pd.NA, inplace=True)

    # Convert columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Impute NaN values with the mean of each column
    df.fillna(df.mean(), inplace=True)

    return df

def get_feature_target(df, target_column):
    """
    Split the DataFrame into features and target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    target_column (str): The name of the target column.

    Returns:
    tuple: A tuple containing the features DataFrame and the target Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y