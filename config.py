import os

class Config:
    # Paths
    INPUT_PATH = 'YOUR_PATH'
    TEST_PATH = 'YOUR PATH'
    TRAIN_CSV_PATH = 'YOUR_PATH'
    BASE_LOCATION = 'YOUR_PATH'
    OUTPUT_FEATURE_PATH = os.path.join(TEST_PATH, 'feat')
    TOKENIZER_PATH = os.path.join(TEST_PATH, 'tokenizer.pkl')
    FINAL_MODEL_PATH = os.path.join(TEST_PATH, 'final_model_with_masked_loss.h5')

    # Video processing
    FRAME_RATE = 3  # Extract one frame every 3 frames
    MAX_FRAMES = 240  # Maximum frames to extract per video
    IMAGE_SIZE = (224, 224)  # Frame resizing dimensions

    # Dataset settings
    MAX_CAPTION_LENGTH = 40  # Maximum tokenized caption length
    VOCAB_SIZE = 10000  # Maximum vocabulary size for tokenizer
    OOV_TOKEN = "<unk>"

    # Model hyperparameters
    EMBEDDING_DIM = 300  # Dimension of embedding layer
    UNITS = 768  # Number of units in LSTM layers
    DROPOUT_RATE = 0.5  # Dropout rate for LSTMs and attention
    KEY_DIM = 128  # Key dimension for multi-head attention
    NUM_HEADS = 8  # Number of attention heads

    # Training settings
    BATCH_SIZE = 16
    EPOCHS = 150
    INITIAL_LEARNING_RATE = 0.0005
    LEARNING_RATE_DECAY_STEPS = 1000
    LEARNING_RATE_DECAY_RATE = 0.96

    # Miscellaneous
    RANDOM_SEED = 42
    SAVE_PLOTS = True  # Save loss plots to file
    LOSS_PLOT_PATH = os.path.join(TEST_PATH, 'loss_plot.png')
    LR_SCHEDULE_PLOT_PATH = os.path.join(TEST_PATH, 'lr_schedule_plot.png')

