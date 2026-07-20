import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    Fill missing values for numerical columns with the median and categorical columns with the mode.
    """
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    for column in data.select_dtypes(include=[object]).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def encode_categorical_features(data):
    """
    Encode categorical features using Label Encoding.
    """
    label_encoders = {}
    for column in data.select_dtypes(include=[object]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def scale_features(data):
    """
    Scale numerical features using Standard Scaler.
    """
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data, scaler

def preprocess_data(file_path):
    """
    Main function to preprocess the data.
    Load data, handle missing values, encode categorical features, and scale numerical features.
    """
    data = load_data(file_path)
    data = handle_missing_values(data)
    data, label_encoders = encode_categorical_features(data)
    data, scaler = scale_features(data)
    return data, label_encoders, scaler