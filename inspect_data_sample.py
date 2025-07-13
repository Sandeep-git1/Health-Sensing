import os

# Define the path to AP01
data_path = r"C:\project_\IIT_Gandhinagar\HealthSense\internship-20250708T091417Z-1-001\internship\Data\AP01"

# Function to read and print first 5 lines of each file
def inspect_files():
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        print(f"\nFile: {file_name}")
        try:
            with open(file_path, 'r') as f:
                lines = [next(f).strip() for _ in range(10)]
                for i, line in enumerate(lines, 1):
                    print(f"Line {i}: {line}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    print("Inspecting sample data from AP01 files:")
    inspect_files()