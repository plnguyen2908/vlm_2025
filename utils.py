import os

def find_json_files(base_path):
    """Recursively find all JSON files in the specified directory."""
    json_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def validate_paths(dataset_path, checkpoint_path):
    """Validate that required paths exist."""
    # Check dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Find JSON files
    json_files = find_json_files(dataset_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")
    
    # Check checkpoint
    checkpoint_exists = os.path.exists(checkpoint_path)
    
    return json_files, checkpoint_exists