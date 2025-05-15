from ultralytics import YOLO

def validate_model():
    # Load the trained model
    model = YOLO('models/best.pt')
    
    # Validate the model
    metrics = model.val(data='data.yaml')
    print(f"Validation metrics: {metrics}")

if __name__ == "__main__":
    validate_model()