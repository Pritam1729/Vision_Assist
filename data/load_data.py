import os
import pandas as pd
import numpy as np

def load_video_features(csv_path, base_location):
    df = pd.read_csv(csv_path)
    video_features = {}
    for _, row in df.iterrows():
        folder = row['folder']
        video_id = row['id']
        video_feature_file = f'{base_location}{folder}/{video_id}.mp4.npy'
        video_features[video_id] = np.load(video_feature_file)
    return df, video_features

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df

