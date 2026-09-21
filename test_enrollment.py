import cv2

from src.database import save_embedding


# ---------------------------------
# 1. Model paths
# ---------------------------------

yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"


# ---------------------------------
# 2. Load models
# ---------------------------------

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


# ---------------------------------
# 3. Load image
# ---------------------------------

image = cv2.imread("test_face.jpg")

if image is None:
    print("❌ Image could not be loaded")
    exit()

print("✓ Image loaded!")


# ---------------------------------
# 4. Detect face
# ---------------------------------

height, width = image.shape[:2]

detector.setInputSize((width, height))

_, faces = detector.detect(image)


if faces is None:
    print("❌ No face detected")
    exit()


print(f"✓ {len(faces)} face(s) detected!")


# ---------------------------------
# 5. Take first face
# ---------------------------------

face = faces[0]


# ---------------------------------
# 6. Align face
# ---------------------------------

aligned_face = recognizer.alignCrop(
    image,
    face
)

print("✓ Face aligned!")


# ---------------------------------
# 7. Generate embedding
# ---------------------------------

embedding = recognizer.feature(
    aligned_face
)

print("✓ Embedding generated!")


# ---------------------------------
# 8. Save embedding
# ---------------------------------

name = "Spoorti"

save_embedding(
    name,
    embedding
)

print(f"✓ {name} successfully enrolled!")
