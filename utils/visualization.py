import matplotlib.pyplot as plt

# Plot training and validation loss over epochs
def plot_training_history(history, save_path=None):
    """
    Plot the training and validation loss over epochs.
    Args:
        history (History object): Keras History object containing training history.
        save_path (str, optional): Path to save the plot as an image. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()

# Function to visualize predictions
def visualize_predictions(video_id, predicted_caption, ground_truth_caption):
    """
    Display the predicted and ground truth captions for a given video.
    Args:
        video_id (str): Identifier of the video being analyzed.
        predicted_caption (str): The caption predicted by the model.
        ground_truth_caption (str): The actual caption from the dataset.
    """
    print(f"Video ID: {video_id}")
    print(f"Predicted Caption: {predicted_caption}")
    print(f"Ground Truth Caption: {ground_truth_caption}")

# Example of visualizing a learning rate schedule
def plot_learning_rate_schedule(lr_schedule, steps=1000, save_path=None):
    """
    Plot the learning rate schedule over a number of steps.
    Args:
        lr_schedule (tf.keras.optimizers.schedules): Learning rate schedule instance.
        steps (int): Number of steps to visualize.
        save_path (str, optional): Path to save the plot as an image. If None, the plot is displayed.
    """
    step_values = range(steps)
    lr_values = [lr_schedule(step) for step in step_values]

    plt.figure(figsize=(10, 6))
    plt.plot(step_values, lr_values, label='Learning Rate')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Learning rate schedule plot saved at: {save_path}")
    else:
        plt.show()

