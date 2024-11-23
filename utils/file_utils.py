import os
import pickle
import numpy as np

# Ensure a directory exists, creating it if necessary
def ensure_directory_exists(directory):
    """
    Ensure that a given directory exists. Create it if it doesn't.
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save an object (e.g., tokenizer) to a file using pickle
def save_pickle(file_path, obj):
    """
    Save an object to a file using pickle.
    Args:
        file_path (str): Path to the file.
        obj (object): Object to save.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

# Load an object (e.g., tokenizer) from a pickle file
def load_pickle(file_path):
    """
    Load an object from a pickle file.
    Args:
        file_path (str): Path to the pickle file.
    Returns:
        object: The loaded object.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Save numpy array to a file
def save_numpy(file_path, array):
    """
    Save a numpy array to a .npy file.
    Args:
        file_path (str): Path to the .npy file.
        array (numpy.ndarray): Array to save.
    """
    np.save(file_path, array)

# Load numpy array from a file
def load_numpy(file_path):
    """
    Load a numpy array from a .npy file.
    Args:
        file_path (str): Path to the .npy file.
    Returns:
        numpy.ndarray: The loaded array.
    """
    return np.load(file_path)

# Check if a file exists
def file_exists(file_path):
    """
    Check if a file exists.
    Args:
        file_path (str): Path to the file.
    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)

# Get a list of all files in a directory with a specific extension
def list_files_with_extension(directory, extension):
    """
    Get a list of all files in a directory with a specific extension.
    Args:
        directory (str): Path to the directory.
        extension (str): File extension (e.g., '.mp4').
    Returns:
        list: List of file paths with the specified extension.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

