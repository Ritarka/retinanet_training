import os

def change_extensions_to_jpg(directory_path):
    # Check if the provided directory path exists
    if not os.path.isdir(directory_path):
        print(f"The provided directory {directory_path} does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Make sure we're working with files, not subdirectories
        if os.path.isfile(file_path):
            # Split filename into name and extension
            file_name, file_extension = os.path.splitext(filename)
            
            # Construct the new filename with .jpg extension
            new_file_path = os.path.join(directory_path, file_name + '.jpg')
            
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed {filename} to {file_name}.jpg")

    print("All file extensions changed to .jpg")

# Usage example
directory = '/home/biometrics/ritarka/data/pure_background/images'
change_extensions_to_jpg(directory)
