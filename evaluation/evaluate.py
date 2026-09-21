import cv2
import pandas as pd


# --------------------------------
# Model paths
# --------------------------------

yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"


# --------------------------------
# Thresholds to test
# --------------------------------

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


# --------------------------------
# Load models
# --------------------------------

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


# --------------------------------
# Generate embedding
# --------------------------------

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load: {image_path}")
        return None

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        print(f"❌ No face detected: {image_path}")
        return None

    face = faces[0]

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    return embedding


# --------------------------------
# Load pairs
# --------------------------------

pairs = pd.read_csv(
    "evaluation/pairs.csv"
)

print(f"✓ Loaded {len(pairs)} evaluation pairs")


# --------------------------------
# Calculate similarity scores
# --------------------------------

results = []


for _, row in pairs.iterrows():

    image1 = row["image1"]
    image2 = row["image2"]
    label = row["label"]

    print(
        f"\nComparing: {image1} ↔ {image2}"
    )

    embedding1 = get_embedding(image1)
    embedding2 = get_embedding(image2)

    if embedding1 is None or embedding2 is None:
        continue

    score = recognizer.match(
        embedding1,
        embedding2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    score = float(score)

    print(f"Similarity: {score:.4f}")
    print(f"Label: {label}")

    results.append({
        "image1": image1,
        "image2": image2,
        "label": label,
        "score": score
    })


results_df = pd.DataFrame(results)


# =================================
# THRESHOLD EVALUATION
# =================================

print("\n")
print("==============================")
print("Threshold Evaluation")
print("==============================")


for threshold in THRESHOLDS:

    true_accepts = 0
    true_rejects = 0
    false_accepts = 0
    false_rejects = 0

    for _, row in results_df.iterrows():

        score = row["score"]
        label = row["label"]

        # Model decision
        accepted = score >= threshold

        # Genuine pair
        if label == "genuine":

            if accepted:
                true_accepts += 1
            else:
                false_rejects += 1

        # Impostor pair
        elif label == "impostor":

            if accepted:
                false_accepts += 1
            else:
                true_rejects += 1

    # Calculate rates

    if true_accepts + false_rejects > 0:
        frr = (
            false_rejects /
            (true_accepts + false_rejects)
        ) * 100
    else:
        frr = 0

    if false_accepts + true_rejects > 0:
        far = (
            false_accepts /
            (false_accepts + true_rejects)
        ) * 100
    else:
        far = 0

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"True Accepts: {true_accepts}"
    )

    print(
        f"True Rejects: {true_rejects}"
    )

    print(
        f"False Accepts: {false_accepts}"
    )

    print(
        f"False Rejects: {false_rejects}"
    )

    print(
        f"FAR: {far:.2f}%"
    )

    print(
        f"FRR: {frr:.2f}%"
    )
