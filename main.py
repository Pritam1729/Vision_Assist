from preprocessing.video_to_frames import video_to_frames
from preprocessing.feature_extraction import model_cnn_load, extract_features
from data.load_data import load_video_features, split_data
from data.tokenizer import create_tokenizer
from model.train_model import train_model
from model.evaluate_model import evaluate_model
import config

def main():
    # Step 1: Preprocess videos
    print("Starting video preprocessing...")
    video_list = [...]  # List of video files
    for video in video_list:
        video_to_frames(video, config.input_path, config.test_path)
    
    # Step 2: Extract features
    print("Extracting features...")
    model = model_cnn_load()
    for video in video_list:
        extract_features(video, model, config.input_path, config.test_path)

    # Step 3: Load data and tokenize
    print("Loading data...")
    df, video_features = load_video_features(config.train_csv_path, config.input_path)
    train_data, val_data, test_data = split_data(df)
    tokenizer = create_tokenizer(train_data['descriptions'])

    # Step 4: Train the model
    print("Training model...")
    train_model(train_data, val_data, video_features, tokenizer)

    # Step 5: Evaluate the model
    print("Evaluating model...")
    evaluate_model(test_data, video_features, tokenizer)

if __name__ == "__main__":
    main()

