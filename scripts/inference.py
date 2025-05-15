from ultralytics import YOLO
import cv2
import os

def run_inference():
    # Load the trained model
    model_path = os.path.join('training_output', 'drone_detector', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    # Create results directory
    results_dir = 'detection_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run inference on test images
    test_dir = os.path.join('drone_dataset_yolo', 'val', 'images')
    for img_name in os.listdir(test_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_dir, img_name)
            
            # Run detection
            results = model.predict(img_path, conf=0.25)
            
            # Save results
            for r in results:
                im_array = r.plot()
                cv2.imwrite(os.path.join(results_dir, f'pred_{img_name}'), im_array)
    
    print(f"Inference completed. Results saved in '{results_dir}' folder")

if __name__ == "__main__":
    run_inference()
