import os
import cv2

def preprocess_images(input_dir, output_dir, img_size=(416, 416)):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(output_dir, img_name), resized_img)
            print(f"Processed {img_name}")

if __name__ == "__main__":
    preprocess_images("../dataset/raw", "../dataset/processed")
