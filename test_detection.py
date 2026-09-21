import cv2

# 1. Location of the YuNet model
model_path = "models/face_detection_yunet_2026may.onnx"

# 2. Load the YuNet face detector
detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

print("✓ YuNet loaded successfully!")

# 3. Read the test image
image = cv2.imread("test_face.jpg")

# Check whether the image was loaded
if image is None:
    print("❌ Could not load test_face.jpg")
    exit()

print("✓ Image loaded successfully!")

# 4. Get image dimensions
height, width = image.shape[:2]

# Tell YuNet the image size
detector.setInputSize((width, height))

# 5. Detect faces
_, faces = detector.detect(image)

# 6. Check the result
if faces is None:
    print("❌ No face detected!")

else:
    print(f"✓ {len(faces)} face(s) detected!")

    # 7. Draw a box around every detected face
    for face in faces:

        x, y, w, h = face[:4]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # 8. Save the result
    cv2.imwrite("detected_faces.jpg", image)

    print("✓ Result saved as detected_faces.jpg")
