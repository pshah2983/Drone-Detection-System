from ultralytics import YOLO
import numpy as np
import os

def evaluate_model():
    # Set project root directory
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Load trained model
    model_path = os.path.join(root_dir, 'training_output', 'drone_detector', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    # Create results directory
    results_dir = os.path.join(root_dir, 'runs')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run validation
    metrics = model.val(
        data='data.yaml',
        split='val',
        imgsz=416,
        batch=8,
        project=results_dir,  # Specify results directory
        name='evaluation'     # Specify run name
    )
    
    print("\nModel Evaluation Metrics:")
    print(f"mAP@0.5: {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    
    # Handle precision and recall as numpy arrays
    precision = metrics.box.p
    recall = metrics.box.r
    
    # Convert to float if array
    p_value = float(np.mean(precision)) if isinstance(precision, np.ndarray) else precision
    r_value = float(np.mean(recall)) if isinstance(recall, np.ndarray) else recall
    
    print(f"Precision: {p_value:.3f}")
    print(f"Recall: {r_value:.3f}")
    print(f"\nResults saved to: {os.path.join(results_dir, 'evaluation')}")

if __name__ == "__main__":
    evaluate_model()