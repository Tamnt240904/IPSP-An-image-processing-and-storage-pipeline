import os
import random
import shutil
import yaml

# Set the root data directory
data_dir = 'train_20241022'  # Root directory containing 'daytime' and 'nighttime'

# Subdirectories for daytime and nighttime data
daytime_dir = os.path.join(data_dir, 'daytime')
nighttime_dir = os.path.join(data_dir, 'nighttime')

# Split ratio
split_ratio = 0.8  # 80% training, 20% validation

# Function to split data into train and valid sets
def split_and_create_yaml(source_dir, output_dir, yaml_path, split_ratio):
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')

    # Create train and valid directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Get all image files
    files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    random.shuffle(files)

    # Calculate split point
    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    valid_files = files[split_point:]

    # Helper function to move files
    def move_files(file_list, destination_dir):
        for file in file_list:
            # Move image
            shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))
            # Move label
            label_file = file.replace('.jpg', '.txt')
            shutil.copy(os.path.join(source_dir, label_file), os.path.join(destination_dir, label_file))

    # Move files to train and valid directories
    move_files(train_files, train_dir)
    move_files(valid_files, valid_dir)

    # Create YAML content
    yaml_content = {
        "names": [
            "Xe máy",
            "Xe ô tô con",
            "Xe vận tải du lịch",
            "Xe vận tải container"
        ],
        "nc": 4,
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(valid_dir)
    }

    # Write YAML to file
    with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(yaml_content, yaml_file, allow_unicode=True)

    print(f"Data split completed for {source_dir}")
    print(f"YAML file created at {yaml_path}")

# Paths for daytime and nighttime
daytime_output_dir = os.path.join(data_dir, 'daytime_split')
nighttime_output_dir = os.path.join(data_dir, 'nighttime_split')

# YAML file paths
daytime_yaml = os.path.join(data_dir, 'daytime.yaml')
nighttime_yaml = os.path.join(data_dir, 'nighttime.yaml')

# Split and create YAML for daytime and nighttime
split_and_create_yaml(daytime_dir, daytime_output_dir, daytime_yaml, split_ratio)
split_and_create_yaml(nighttime_dir, nighttime_output_dir, nighttime_yaml, split_ratio)
