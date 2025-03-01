import cv2
import dlib
import numpy as np

# Load facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the accessory image (sunglasses)
accessory = cv2.imread("C:/Users/Aissa/OneDrive/Documents/GitHub/68_points/sunglasses1.png", -1)

# Ensure accessory is loaded correctly
if accessory is None:
    print("❌ Error: Could not load sunglasses image. Check the file path!")
    exit()

# Check if the image has an alpha channel, convert if necessary
if accessory.shape[2] == 3:  
    print("⚠️ No alpha channel detected! Converting to RGBA...")
    accessory = cv2.cvtColor(accessory, cv2.COLOR_BGR2BGRA)

# Function to overlay accessory
def overlay_accessory(image, accessory, position):
    x, y, w, h = position
    accessory = cv2.resize(accessory, (w, h), interpolation=cv2.INTER_AREA)

    if accessory.shape[2] == 4:
        alpha_s = accessory[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    else:
        print("⚠️ Warning: No alpha channel detected! Using basic overlay.")
        alpha_s = 1
        alpha_l = 0

    for c in range(0, 3):
        image[y:y+h, x:x+w, c] = (alpha_s * accessory[:, :, c] + alpha_l * image[y:y+h, x:x+w, c])

    return image

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get landmark points for positioning the sunglasses
        left_eye_x = landmarks.part(36).x  # Outer left eye corner
        right_eye_x = landmarks.part(45).x  # Outer right eye corner
        eye_center_y = (landmarks.part(36).y + landmarks.part(45).y) // 2  # Midpoint between eyes

        # Compute sunglasses width based on eye width
        width = int((right_eye_x - left_eye_x) * 1.8)  # Adjust width scale
        height = int(width * 0.5)  # Maintain aspect ratio

        # Adjust position to sit properly on eyes
        x1 = left_eye_x - int(width * 0.15)  # Shift left slightly
        y1 = eye_center_y - int(height * 0.6)  # Move up to align with eyebrows

        position = (x1, y1, width, height)

        # Overlay sunglasses on the face
        frame = overlay_accessory(frame, accessory, position)

    cv2.imshow("Virtual Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
