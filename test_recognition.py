import cv2
import numpy as np
from src.database import load_database


# -----------------------------
# 1. Model paths
# -----------------------------
yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"


# Temporary threshold for testing
THRESHOLD = 0.40


# -----------------------------
# 2. Load models
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
# 3. Generate embedding
# -----------------------------
def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image could not be loaded")
        return None

    print("✓ Image loaded!")

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        print("❌ No face detected")
        return None

    print(f"✓ {len(faces)} face(s) detected!")

    # Use the first detected face
    face = faces[0]

    # Align face
    aligned_face = recognizer.alignCrop(image, face)

    print("✓ Face aligned!")

    # Generate embedding
    embedding = recognizer.feature(aligned_face)

    print("✓ Embedding generated!")

    return embedding


# -----------------------------
# 4. Create query embedding
# -----------------------------
query_embedding = get_embedding("test_other.jpg")

if query_embedding is None:
    exit()


# -----------------------------
# 5. Load database
# -----------------------------
database = load_database()

if not database:
    print("❌ Database is empty")
    exit()

print("✓ Database loaded!")


# -----------------------------
# 6. Compare with database
# -----------------------------
best_name = "UNKNOWN"
best_score = -1


for name, stored_embeddings in database.items():

    person_best_score = -1

    for stored_embedding in stored_embeddings:

        # Convert JSON list back to NumPy array
        stored_embedding = np.array(
            stored_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        # Calculate cosine similarity
        score = recognizer.match(
            query_embedding,
            stored_embedding,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        score = float(score)

        # Keep best score for this person
        if score > person_best_score:
            person_best_score = score

    print(f"{name}: {person_best_score:.4f}")

    # Keep best person
    if person_best_score > best_score:
        best_score = person_best_score
        best_name = name


# -----------------------------
# 7. Apply threshold
# -----------------------------
print("\n-----------------------------")
print(f"Best match: {best_name}")
print(f"Similarity: {best_score:.4f}")
print(f"Threshold: {THRESHOLD}")
print("-----------------------------")


if best_score >= THRESHOLD:

    print(f"✓ Identity: {best_name}")
    print("✓ Status: KNOWN")

else:

    print("❌ Identity: UNKNOWN")
    print("❌ Status: UNKNOWN / REJECTED")
