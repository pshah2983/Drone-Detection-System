import os
import shutil
from sklearn.model_selection import train_test_split
import glob

def organize_dataset():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    source_dir = os.path.join(base_dir, 'drone_dataset_yolo')
    
    print(f"Looking for images in: {source_dir}")
    
    # Create necessary directories
    train_img_dir = os.path.join(source_dir, 'train', 'images')
    train_label_dir = os.path.join(source_dir, 'train', 'labels')
    val_img_dir = os.path.join(source_dir, 'val', 'images')
    val_label_dir = os.path.join(source_dir, 'val', 'labels')
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Search recursively for image files
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    if not image_files:
        print("No image files found! Please check the source directory structure.")
        print("Expected structure:")
        print(f"{source_dir}/")
        print("    ├── images/")
        print("    │   └── *.jpg/png")
        print("    └── labels/")
        print("        └── *.txt")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Get all label files
    label_files = glob.glob(os.path.join(source_dir, '**', '*.txt'), recursive=True)
    print(f"Found {len(label_files)} label files")
    
    # Split into train and validation sets (80/20)
    train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Counter for successful copies
    train_copied = 0
    val_copied = 0
    
    # Move files to respective directories
    for img_path in train_images:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        
        # Move image
        shutil.copy2(img_path, os.path.join(train_img_dir, img_filename))
        
        # Find and move corresponding label
        possible_label_paths = [
            os.path.join(os.path.dirname(img_path), '..', 'labels', label_filename),
            os.path.join(os.path.dirname(img_path), label_filename)
        ]
        
        for label_path in possible_label_paths:
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(train_label_dir, label_filename))
                train_copied += 1
                break
    
    for img_path in val_images:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        
        # Move image
        shutil.copy2(img_path, os.path.join(val_img_dir, img_filename))
        
        # Find and move corresponding label
        possible_label_paths = [
            os.path.join(os.path.dirname(img_path), '..', 'labels', label_filename),
            os.path.join(os.path.dirname(img_path), label_filename)
        ]
        
        for label_path in possible_label_paths:
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(val_label_dir, label_filename))
                val_copied += 1
                break
    
    print("\nDataset organization completed!")
    print(f"Training set: {len(train_images)} images, {train_copied} labels")
    print(f"Validation set: {len(val_images)} images, {val_copied} labels")

if __name__ == "__main__":
    organize_dataset()