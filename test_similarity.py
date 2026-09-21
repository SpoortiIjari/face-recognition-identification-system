import cv2


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
# 3. Function to generate embedding
# ---------------------------------

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load {image_path}")
        return None

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        print(f"❌ No face detected in {image_path}")
        return None

    print(f"✓ Face detected in {image_path}")

    # Take first detected face
    face = faces[0]

    # Align face
    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    # Generate embedding
    embedding = recognizer.feature(
        aligned_face
    )

    return embedding


# ---------------------------------
# 4. Generate embeddings
# ---------------------------------

embedding1 = get_embedding("test_face.jpg")

embedding2 = get_embedding("test_face_2.jpg")


# ---------------------------------
# 5. Compare embeddings
# ---------------------------------

if embedding1 is not None and embedding2 is not None:

    score = recognizer.match(
        embedding1,
        embedding2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    print("\n==============================")
    print("FACE SIMILARITY RESULT")
    print("==============================")


# -----------------------------
# 1. Model paths
# -----------------------------

yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"


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
# 3. Function to generate embedding
# -----------------------------

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load {image_path}")
        return None

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        print(f"❌ No face detected in {image_path}")
        return None

    print(f"✓ Face detected in {image_path}")

    face = faces[0]

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    return embedding


# -----------------------------
# 4. Generate embeddings
# -----------------------------

embedding1 = get_embedding("test_face.jpg")

embedding2 = get_embedding("test_face_2.jpg")

embedding_other = get_embedding("test_other.jpg")


# -----------------------------
# 5. Compare faces
# -----------------------------

if (
    embedding1 is not None
    and embedding2 is not None
    and embedding_other is not None
):

    # Your photo vs your other photo
    score_same = recognizer.match(
        embedding1,
        embedding2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    # Your photo vs another person's photo
    score_different = recognizer.match(
        embedding1,
        embedding_other,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    # -----------------------------
    # 6. Display results
    # -----------------------------

    print("\n==============================")
    print("FACE SIMILARITY EXPERIMENT")
    print("==============================")

    print(
        f"\nYour photo ↔ Your photo: "
        f"{score_same:.4f}"
    )

    print(
        f"Your photo ↔ Other person: "
        f"{score_different:.4f}"
    )
    print(f"Cosine similarity: {score:.4f}")
