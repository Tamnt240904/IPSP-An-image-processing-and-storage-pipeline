import os
path = "original_data"
nighttime_folder = os.path.join(path, 'nighttime')

class_id_mapping = {4: 0, 5: 1, 6: 2, 7: 3}

for filename in os.listdir(nighttime_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(nighttime_folder, filename)

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the class IDs in each line
        modified_lines = []
        for line in lines:
            parts = line.strip().split(' ')
            class_id = int(parts[0])  # Get the class ID (first element in the line)

            # If the class ID needs to be changed according to the mapping
            if class_id in class_id_mapping:
                new_class_id = class_id_mapping[class_id]
                parts[0] = str(new_class_id)  # Replace with the new class ID

            # Recreate the modified line and append to list
            modified_lines.append(' '.join(parts) + '\n')

        # Overwrite the original file with the modified lines
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

print("Class IDs in 'nighttime' folder have been updated successfully.")