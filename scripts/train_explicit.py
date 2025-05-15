from ultralytics import YOLO
import os

def train_model():
    # Create explicit save directory
    save_dir = os.path.join(os.getcwd(), 'training_output')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = YOLO('models/yolov8n.pt')
    
    try:
        results = model.train(
            data='data.yaml',
            epochs=1,
            imgsz=416,
            batch=8,
            name='drone_detector',
            project=save_dir,
            exist_ok=True,
            patience=20
        )
        print(f"Training completed. Model saved in {save_dir}")
    except Exception as e:
        print(f"Training error: {e}")

if __name__ == "__main__":
    train_model()