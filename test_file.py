import os

file_path = r"C:\Users\Aissa\OneDrive\Documents\GitHub\68_points\sunglasses.png"

if os.path.exists(file_path):
    print("✅ File exists!")
else:
    print("❌ File NOT found. Check the path.")
