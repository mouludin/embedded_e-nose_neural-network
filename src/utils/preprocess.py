import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def clean_csv(file_path):
    """Membersihkan file CSV dari baris yang tidak konsisten"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    num_columns = len(lines[0].strip().split(','))
    clean_lines = [lines[0]]  # Header
    for line in lines[1:]:
        if len(line.strip().split(',')) == num_columns:
            clean_lines.append(line)
    
    clean_path = file_path.replace('.csv', '_clean.csv')
    with open(clean_path, 'w') as f:
        f.writelines(clean_lines)
    return clean_path

def load_and_merge_data(dataset_folder):
    """Muat dan gabungkan semua file CSV dalam folder"""
    all_dataframes = []
    sample_labels = []
    
    for f in os.listdir(dataset_folder):
        if f.endswith('.csv'):
            file_path = clean_csv(os.path.join(dataset_folder, f))
            df = pd.read_csv(file_path).drop('Timestamp', axis=1)
            filename = os.path.splitext(f)[0]
            df['source_file'] = filename
            all_dataframes.append(df)
            sample_labels.extend([filename] * len(df))
    
    return pd.concat(all_dataframes, ignore_index=True), sorted(list(set(sample_labels)))

def preprocess_data(data, class_names):
    """Normalisasi dan one-hot encoding"""
    # One-hot encoding
    y = np.array([[1 if name == cls else 0 for cls in class_names] 
                 for name in data['source_file']], dtype=np.float32)
    
    # Normalisasi fitur
    X = data.drop('source_file', axis=1).values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler