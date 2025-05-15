import os

def find_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    possible_locations = [
        'models',
        'runs/detect/drone_detection_model/weights',
        'trained_models',
        '.'
    ]
    
    print("Searching for model files...")
    for location in possible_locations:
        full_path = os.path.join(base_dir, location)
        if os.path.exists(full_path):
            for file in os.listdir(full_path):
                if file.endswith('.pt'):
                    print(f"Found model: {os.path.join(full_path, file)}")

if __name__ == "__main__":
    find_model()