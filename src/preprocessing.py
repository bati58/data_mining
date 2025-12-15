import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self, file_path):
        """Load data from CSV file"""
        df = pd.read_csv(file_path)
        return df
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values with specified strategy"""
        if strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        elif strategy == 'drop':
            df = df.dropna()
        
        return df
    
    def encode_categorical(self, df, columns=None):
        """Encode categorical variables"""
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        return df
    
    def scale_features(self, df, columns=None):
        """Scale numerical features"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_scaled = df.copy()
        df_scaled[columns] = self.scaler.fit_transform(df[columns])
        
        return df_scaled