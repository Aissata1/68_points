import cv2

file_path = r"C:\Users\Aissa\OneDrive\Documents\GitHub\68_points\sunglasses1.png"

accessory = cv2.imread(file_path, -1)

if accessory is None:
    print("❌ OpenCV could not load the image. Check the file path or image format!")
else:
    print(f"✅ Image loaded! Shape: {accessory.shape}")  # Print image shape
