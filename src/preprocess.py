import os
import pandas as pd
import numpy as np

def clean_and_prepare_data(raw_file_path, output_file_path):
    print(f"Loading raw data from {raw_file_path}...")
    
    columns_to_keep = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
        'home_ownership', 'annual_inc', 'purpose', 'dti', 
        'revol_util', 'total_acc', 'issue_d', 'earliest_cr_line', 'loan_status'
    ]
    
    df = pd.read_csv(raw_file_path, usecols=columns_to_keep, low_memory=False)
    
    print("Filtering for finished loans...")
    valid_statuses = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    df['target'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    df = df.drop(columns=['loan_status'])

    print("Engineering features...")
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
        
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
        df['credit_hist_months'] = (df['issue_d'] - df['earliest_cr_line']).dt.days // 30
        df = df.drop(columns=['issue_d', 'earliest_cr_line'])

    print("Imputing missing values...")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    print("Encoding categorical variables...")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print("Optimizing memory footprint...")
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype(np.int32)
    
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(np.uint8)

    print(f"Saving final ML-ready dataset to {output_file_path}")
    df.to_pickle(output_file_path)
    print(f"Done! Final shape: {df.shape}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    
    raw_filename = "accepted_2007_to_2018Q4.csv.gz" 
    
    raw_file = os.path.join(data_folder, raw_filename)
    clean_file = os.path.join(data_folder, "ml_ready_lending_club.pkl")
    
    clean_and_prepare_data(raw_file, clean_file)