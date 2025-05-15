from ultralytics import YOLO
import cv2
import os

def process_video(input_video=None, use_webcam=False):
    # Load model
    model_path = os.path.join('training_output', 'drone_detector', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    # Set up video capture
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        if input_video is None:
            raise ValueError("Please provide input video path")
        cap = cv2.VideoCapture(input_video)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    cv2.namedWindow('Drone Detection', cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run inference on frame
        results = model.predict(
            source=frame,
            conf=0.45,
            show=False,  # Don't show automatically
            stream=True  # Enable streaming mode
        )
        
        # Process results
        for r in results:
            # Plot results on frame
            annotated_frame = r.plot()
            
            # Display the frame
            cv2.imshow('Drone Detection', annotated_frame)
            
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # For webcam
    #process_video(use_webcam=True)
    
    # For video file
    process_video("drone testing video.mp4")