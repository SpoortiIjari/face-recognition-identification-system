import cv2

# -----------------------------
# 1. Model paths
# -----------------------------
yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"


# -----------------------------
# 2. Load the models
# -----------------------------
detector = cv2.FaceDetectorYN.create(
    yunet_model,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    sface_model,
    ""
)

print("✓ Models loaded!")


# -----------------------------
# 3. Load the image
# -----------------------------
image = cv2.imread("test_face.jpg")

if image is None:
    print("❌ Could not load test_face.jpg")
    exit()

print("✓ Image loaded!")


# -----------------------------
# 4. Detect the face
# -----------------------------
height, width = image.shape[:2]

detector.setInputSize((width, height))

_, faces = detector.detect(image)

if faces is None:
    print("❌ No face detected!")
    exit()

print(f"✓ {len(faces)} face(s) detected!")


# -----------------------------
# 5. Select the first face
# -----------------------------
face = faces[0]


# -----------------------------
# 6. Align and crop the face
# -----------------------------
aligned_face = recognizer.alignCrop(
    image,
    face
)

print("✓ Face aligned!")


# -----------------------------
# 7. Generate embedding
# -----------------------------
embedding = recognizer.feature(
    aligned_face
)

print("✓ Face embedding generated!")


# -----------------------------
# 8. Display embedding information
# -----------------------------
print("Embedding shape:", embedding.shape)

print("\nFirst 10 embedding values:")
print(embedding[0][:10])
