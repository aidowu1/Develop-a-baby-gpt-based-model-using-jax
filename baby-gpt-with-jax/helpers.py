import pickle


def save_object(obj, file_path):
    """Save a Python object to a file using pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_object(file_path):
    """Load a Python object from a file using pickle."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
        